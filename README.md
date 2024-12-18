Install these libraries using pip commands first

ollama
streamlit
gradio
pdfplumber 
langchain
langchain-core
langchain-ollama
langchain_community
langchain_text_splitters
unstructured
unstructured[all-docs]
onnx==1.17.0
protobuf==3.20.3
chromadb==0.4.22
Pillow
PyPdf
numpy


To implement the models,
first run these commands
1)ollama pull llama2
2)ollama pull nomic-embed-text

next, download this repository as ZIP file or clone this repository using git

change your working directory to "ollama_pdf_rag" using the cd command

Next , 
create a virtual environment (python -m venv venv)
and run ".\venv\Scripts\activate" (for windows)
install all the dependencies mentioned above

With no further errors,
Execute the streamlit application using the following command

streamlit run streamlit_ap.py


UI:

upload a pdf or select the sample one which is already present
input your query and it gives you the answer

Note:It may take a while to load the models so please be patient! : )

Happy coding----:) Mani Vaishnavi L

