def schoolbook_2x2x1(i_off, j_off, k_idx):
    cols = []
    for local_j in range(2):
        U_dict = {(i_off, j_off + local_j): 1, (i_off+1, j_off + local_j): 1}
        V_dict = {(j_off + local_j, k_idx): 1}
        W_dict = {(i_off, k_idx): 1, (i_off+1, k_idx): 1}
        cols.append(add_term(U_dict, V_dict, W_dict))
    return cols
