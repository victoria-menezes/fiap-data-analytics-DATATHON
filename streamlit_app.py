import pandas as pd
import streamlit as st
import numpy as np
import joblib
import os

DEBUG = False

# if running locally, run via cmd / powershell and not vs code terminal (version inconsistencies)

if DEBUG:
    import sklearn
    import sys
    
    st.write(f"Python: {sys.version}")
    st.write(f"Joblib: {joblib.__version__}")
    st.write(f"Sklearn: {sklearn.__version__}") 

PATH_JOBLIB = 'model'
FILE_JOBLIB = os.path.join(PATH_JOBLIB, 'model.joblib')
FILE_JOBLIB_THRESHOLD = os.path.join(PATH_JOBLIB, 'threshold')

threshold = 0.5
with open(FILE_JOBLIB_THRESHOLD, 'r') as f:
    threshold = float(f.read())

features = [
		'inde', 'fase', 'ian', 'ida', 'ieg', 'iaa', 'ips', 'ipp', 'ipv',
		'nota_mat', 'nota_port', 'nota_ing',
        'idade', 'anos_estudados',
		'inde_diff', 'ian_diff', 'ida_diff', 'ieg_diff',
		'iaa_diff', 'ips_diff', 'ipp_diff', 'ipv_diff',
	 	'genero', 'pedra'
]

target = 'defasará'
model = joblib.load(FILE_JOBLIB)

user_data = {}

def write_header(text, h_level : int = 2):
    prefix = '#'*h_level
    st.write(f'{prefix} {text}')

def count_to_coord(count, ncols : int = 3):
    col = count % ncols
    row = count // ncols
    
    return row, col

### Website
st.title("Análise e previsão de defasagem dos alunos Passos Mágicos")

write_header('Modelo preditivo')
st.write('Preencha os dados do aluno abaixo, e no final aperte o botão "enviar". O modelo irá avaliar o risco de defasagem do aluno e retornar com uma resposta binária, além da probabilidade calculada.')
st.write(f'##### Foi considerado qualquer aluno com uma probabilidade de defasagem maior que **{threshold*100:.2f}%** como um aluno em risco. Esse ponto de referência foi calculado com base do f1-score do modelo.')
st.write(f'O modelo foi construindo visando minimizar os **falsos negativos**; ou seja, **sinalizar o máximo de alunos com defasagem possível**, afim de evitar que algum deles passe despercebido.\nComo consequência, o modelo sinaliza **falsos positivos** com uma frequência mais elevada. As estatísticas de precision, recall, etc, estão disponíveis no notebook do github.')
st.write('Caso um dado ultrapassar limites impostos pelo app, deixe o valor no máximo (caso maior), ou no mínimo (caso menor).')
st.write('O modelo irá tentar preencher dados faltantes, mas ele será mais efetivo com mais informação.')
st.write('O relatório com a análise dos dados forcenidos está disponível no github.')
st.link_button(label='GitHub', url = 'https://github.com/victoria-menezes/fiap-data-analytics-DATATHON')


write_header('Dados Pessoais', 3)
ncols = 3
cols = []
cols = st.columns(ncols)
count = 0

current_col = cols[count_to_coord(count, ncols)[1]]
user_data['idade'] = current_col.number_input('Idade', min_value = 7, max_value = 19, value = None, step=1)
current_col.write('Qual a idade do aluno?')
count +=1

current_col = cols[count_to_coord(count, ncols)[1]]
user_data['anos_estudados'] = current_col.number_input('Anos estudados', min_value = 0, max_value = 8, value = None, step = 1)
current_col.write('Há quantos anos o aluno estuda com a passos mágicos?')
count +=1

current_col = cols[count_to_coord(count, ncols)[1]]
user_data['genero'] = current_col.selectbox('Gênero', options = ['', 'Feminino', 'Masculino'], index=0)
current_col.write('Qual o gênero do aluno?')

gender_map = {
    'Feminino' : 'Feminino',
    'Masculino' : 'Masculino',
    '': np.nan
}
user_data['genero'] = gender_map[user_data['genero']]
count +=1

write_header('Indicadores do ano escolar mais recente', 3)
ncols = 2
cols = []
cols = st.columns(ncols)
count = 0

current_col = cols[count_to_coord(count, ncols)[1]]
ind = 'pedra'
user_data[ind] = current_col.selectbox(ind.capitalize(), options = ['Quartzo', 'Ágata', 'Ametista', 'Topázio'], index=0)
pedra_map = {
    'Quartzo' : 'Quartzo',
    'Ágata' : 'Ágata',
    'Ametista': 'Ametista',
    'Topázio':'Topázio',
    '': np.nan
}
user_data[ind] = pedra_map[user_data[ind]]
count +=1

current_col = cols[count_to_coord(count, ncols)[1]]
ind = 'fase'
user_data['fase'] = current_col.number_input(ind.capitalize(), min_value = 0, max_value = 8, value = None, step = 1)
current_col.write('Considere 0 como a fase de alfabetização')
count +=1

write_header('Indicadores do ano escolar mais recente', 3)
ncols = 3
cols = []
cols = st.columns(ncols)
count = 0

current_col = cols[count_to_coord(count, ncols)[1]]
ind = 'inde'
user_data[ind] = current_col.number_input(ind.upper(), min_value = 0.0, max_value = 10.0, value = None, step = 0.1)
count +=1

current_col = cols[count_to_coord(count, ncols)[1]]
ind = 'ian'
user_data[ind] = current_col.selectbox(ind.upper(), options = ['', '2.5', '5.0', '10.0'], index=0)
ian_map = {
    '2.5' : 2.5,
    '5.0' : 5,
    '10.0': 10,
    '': np.nan
}
user_data[ind] = ian_map[user_data[ind]]
count +=1

current_col = cols[count_to_coord(count, ncols)[1]]
ind = 'ida'
user_data[ind] = current_col.number_input(ind.upper(), min_value = 0.0, max_value = 10.0, value = None, step = 0.1)
count +=1

current_col = cols[count_to_coord(count, ncols)[1]]
ind = 'ieg'
user_data[ind] = current_col.number_input(ind.upper(), min_value = 0.0, max_value = 10.0, value = None, step = 0.1)
count +=1

current_col = cols[count_to_coord(count, ncols)[1]]
ind = 'iaa'
user_data[ind] = current_col.number_input(ind.upper(), min_value = 0.0, max_value = 10.0, value = None, step = 0.1)
count +=1

current_col = cols[count_to_coord(count, ncols)[1]]
ind = 'ips'
user_data[ind] = current_col.number_input(ind.upper(), min_value = 0.0, max_value = 10.0, value = None, step = 0.1)
count +=1

current_col = cols[count_to_coord(count, ncols)[1]]
ind = 'ipp'
user_data[ind] = current_col.number_input(ind.upper(), min_value = 0.0, max_value = 10.0, value = None, step = 0.1)
count +=1

current_col = cols[count_to_coord(count, ncols)[1]]
ind = 'ipv'
user_data[ind] = current_col.number_input(ind.upper(), min_value = 0.0, max_value = 10.0, value = None, step = 0.1)
count +=1

write_header('Notas do ano escolar mais recente', 3)
ncols = 3
cols = []
cols = st.columns(ncols)
count = 0

current_col = cols[count_to_coord(count, ncols)[1]]
ind = 'nota_port'
user_data[ind] = current_col.number_input('Português', min_value = 0.0, max_value = 10.0, value = None, step = 0.1)
count +=1

current_col = cols[count_to_coord(count, ncols)[1]]
ind = 'nota_mat'
user_data[ind] = current_col.number_input('Matemática', min_value = 0.0, max_value = 10.0, value = None, step = 0.1)
count +=1

current_col = cols[count_to_coord(count, ncols)[1]]
ind = 'nota_ing'
user_data[ind] = current_col.number_input('Inglês', min_value = 0.0, max_value = 10.0, value = None, step = 0.1)
count +=1


write_header('Indicadores do ano escolar anterior', 3)
st.write('Dados de um ano anterior aos dados preenchidos acima.')
ncols = 3
cols = []
cols = st.columns(ncols)
count = 0

past_idx = {}

current_col = cols[count_to_coord(count, ncols)[1]]
ind = 'inde'
past_idx[ind] = current_col.number_input(f'{ind.upper()} passado', min_value = 0.0, max_value = 10.0, value = None, step = 0.1)
count +=1

current_col = cols[count_to_coord(count, ncols)[1]]
ind = 'ian'
past_idx[ind] = current_col.selectbox(f'{ind.upper()} passado', options = ['', '2.5', '5.0', '10.0'], index=0)
past_idx[ind] = ian_map[past_idx[ind]]
count +=1

current_col = cols[count_to_coord(count, ncols)[1]]
ind = 'ida'
past_idx[ind] = current_col.number_input(f'{ind.upper()} passado', min_value = 0.0, max_value = 10.0, value = None, step = 0.1)
count +=1

current_col = cols[count_to_coord(count, ncols)[1]]
ind = 'ieg'
past_idx[ind] = current_col.number_input(f'{ind.upper()} passado', min_value = 0.0, max_value = 10.0, value = None, step = 0.1)
count +=1

current_col = cols[count_to_coord(count, ncols)[1]]
ind = 'iaa'
past_idx[ind] = current_col.number_input(f'{ind.upper()} passado', min_value = 0.0, max_value = 10.0, value = None, step = 0.1)
count +=1

current_col = cols[count_to_coord(count, ncols)[1]]
ind = 'ips'
past_idx[ind] = current_col.number_input(f'{ind.upper()} passado', min_value = 0.0, max_value = 10.0, value = None, step = 0.1)
count +=1

current_col = cols[count_to_coord(count, ncols)[1]]
ind = 'ipp'
past_idx[ind] = current_col.number_input(f'{ind.upper()} passado', min_value = 0.0, max_value = 10.0, value = None, step = 0.1)
count +=1

current_col = cols[count_to_coord(count, ncols)[1]]
ind = 'ipv'
past_idx[ind] = current_col.number_input(f'{ind.upper()} passado', min_value = 0.0, max_value = 10.0, value = None, step = 0.1)
count +=1

for ind in past_idx:
    diff_name = f'{ind}_diff'
    current = user_data[ind]

    if (current == np.nan) or (current == None) or (past_idx[ind] == np.nan) or (past_idx[ind] == None):
        user_data[diff_name] = None
    else:
        user_data[diff_name] = current - past_idx[ind]


# formatting missing values to np.nan
# data = {np.nan if i == None else i for i in user_data}

for data in user_data:
    if user_data[data] == None:
        user_data[data] = np.nan
    if isinstance(user_data[data], int):
        user_data[data] = float(user_data[data])
        
if DEBUG:
    st.write(user_data)
    st.write(isinstance(user_data, dict))
    for data in user_data:
        st.write(isinstance(user_data[data], float))

data = [user_data[feat] for feat in features]
user_df = pd.DataFrame(
    [data],
    columns = features
)
if DEBUG:
    st.write(user_df)

if st.button("Enviar"):

    # user_df = utils.apply_pipeline_ML(user_df)
    # st.write(user_df.columns)
    # user_df = user_df[features]
    proba = model.predict_proba(user_df)[:, -1][0]
    if DEBUG:
        st.write(proba)
    if proba > threshold:
        st.error(f"### Esse aluno apresenta indícios de defasagem futura\n#### Probabilidade calculada: {proba*100:.2f}%")
    else:
        st.success(f"### Esse aluno não apresenta indícios de defasagem futura\n#### Probabilidade calculada: {proba*100:.2f}%")