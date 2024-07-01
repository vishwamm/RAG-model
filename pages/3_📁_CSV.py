import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
from langchain_community.document_loaders.csv_loader import CSVLoader
load_dotenv()
groq_api_key=os.environ['GROQ_API_KEY']
embeddings=OllamaEmbeddings()
llm=ChatGroq(groq_api_key=groq_api_key,
             model_name="llama3-70b-8192")
prompt=ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}

"""
)

st.text("Give me a prompt:")
csv_prompt=st.text_input("Enter the prompt to search")
with st.spinner("Loading csv file"):
    csv_loader = CSVLoader("first.csv")
    csv_docs=csv_loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    csv_final_documents=text_splitter.split_documents(csv_docs[:50])
    csv_vectors=FAISS.from_documents(csv_final_documents,embeddings)
document_chain = create_stuff_documents_chain(llm, prompt)
csv_retriever = csv_vectors.as_retriever()
csv_retrieval_chain = create_retrieval_chain(csv_retriever, document_chain)
if csv_prompt:
    with st.spinner("generating the response"):
        response=csv_retrieval_chain.invoke({"input":csv_prompt})
st.write(response['answer'])