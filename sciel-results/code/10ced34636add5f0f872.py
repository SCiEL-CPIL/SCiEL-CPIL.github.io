import time
import numpy as np
from itertools import product

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 3, 4, 5
    
    def add_term(U_dict, V_dict, W_dict):
        U_vec = np.zeros(n*m, dtype=int)
        V_vec = np.zeros(m*p, dtype=int)
        W_vec = np.zeros(p*n, dtype=int)
        for (i,j) in U_dict:
            U_vec[i*m + j] = U_dict[(i,j)]
        for (j,k) in V_dict:
            V_vec[j*p + k] = V_dict[(j,k)]
        for (i,k) in W_dict:
            W_vec[k*n + i] = W_dict[(i,k)]
        return U_vec, V_vec, W_vec

    def strassen_2x2x2(i_off, j_off, k_off):
        cols = []
        U1 = {(i_off, j_off): 1, (i_off+1, j_off+1): 1}
        V1 = {(j_off, k_off): 1, (j_off+1, k_off+1): 1}
        W1 = {(i_off, k_off): 1, (i_off+1, k_off+1): 1}
        cols.append(add_term(U1, V1, W1))
        U2 = {(i_off+1, j_off): 1, (i_off+1, j_off+1): 1}
        V2 = {(j_off, k_off): 1}
        W2 = {(i_off+1, k_off): 1, (i_off+1, k_off+1): -1}
        cols.append(add_term(U2, V2, W2))
        U3 = {(i_off, j_off): 1}
        V3 = {(j_off+1, k_off): 1, (j_off+1, k_off+1): -1}
        W3 = {(i_off, k_off+1): 1, (i_off+1, k_off+1): 1}
        cols.append(add_term(U3, V3, W3))
        U4 = {(i_off+1, j_off+1): 1}
        V4 = {(j_off, k_off): 1, (j_off, k_off+1): -1}
        W4 = {(i_off, k_off): 1, (i_off+1, k_off): 1}
        cols.append(add_term(U4, V4, W4))
        U5 = {(i_off, j_off): 1, (i_off, j_off+1): 1}
        V5 = {(j_off+1, k_off+1): 1}
        W5 = {(i_off, k_off): -1, (i_off, k_off+1): 1}
        cols.append(add_term(U5, V5, W5))
        U6 = {(i_off+1, j_off): 1, (i_off, j_off): -1}
        V6 = {(j_off, k_off): 1, (j_off, k_off+1): 1}
        W6 = {(i_off+1, k_off+1): 1}
        cols.append(add_term(U6, V6, W6))
        U7 = {(i_off, j_off+1): 1, (i_off+1, j_off+1): -1}
        V7 = {(j_off+1, k_off): 1, (j_off+1, k_off+1): 1}
        W7 = {(i_off, k_off): 1, (i_off+1, k_off): -1}
        cols.append(add_term(U7, V7, W7))
        return cols

    def schoolbook_block(i_set, j_set, k_set):
        cols = []
        for i in i_set:
            for j in j_set:
                for k in k_set:
                    U_dict = {(i, j): 1}
                    V_dict = {(j, k): 1}
                    W_dict = {(i, k): 1}
                    cols.append(add_term(U_dict, V_dict, W_dict))
        return cols

    def try_decomposition(row_splits, col_splits):
        """Try a decomposition based on row and column splits.
        row_splits: list of tuples (row_set, col_split) where col_split is a list of col_sets
        """
        columns = []
        for row_set, col_split in row_splits:
            for j_set in col_split:
                for k_set in col_split:
                    if len(row_set) == 2 and len(j_set) == 2 and len(k_set) == 2:
                        columns.extend(strassen_2x2x2(min(row_set), min(j_set), min(k_set)))
                    else:
                        columns.extend(schoolbook_block(row_set, j_set, k_set))
        return columns

    # Try multiple block partitionings
    best_columns = None
    best_rank = n * m * p  # schoolbook upper bound

    # Partitioning 1: Current approach
    # Rows {0,1} with cols split as [{0,1},{2,3}], rows {2} with cols {0,1,2,3,4}
    cols1 = []
    cols1.extend(strassen_2x2x2(0, 0, 0))
    cols1.extend(strassen_2x2x2(0, 2, 0))
    cols1.extend(schoolbook_block([0,1], [0,1], [2,3,4]))
    cols1.extend(schoolbook_block([0,1], [2,3], [2,3,4]))
    cols1.extend(schoolbook_block([2], [0,1,2,3], [0,1,2,3,4]))
    if len(cols1) < best_rank:
        best_rank = len(cols1)
        best_columns = cols1

    # Partitioning 2: Rows {0,1} with cols split as [{0,1,2},{3,4}]
    cols2 = []
    cols2.extend(strassen_2x2x2(0, 0, 0))
    cols2.extend(strassen_2x2x2(0, 2, 0))
    cols2.extend(schoolbook_block([0,1], [0,1,2], [3,4]))
    cols2.extend(schoolbook_block([0,1], [3], [0,1,2,3,4]))
    cols2.extend(schoolbook_block([2], [0,1,2,3], [0,1,2,3,4]))
    if len(cols2) < best_rank:
        best_rank = len(cols2)
        best_columns = cols2

    # Partitioning 3: Try different row grouping
    # Rows {0,2} with Strassen, row {1} schoolbook
    cols3 = []
    cols3.extend(strassen_2x2x2(0, 0, 0))
    cols3.extend(strassen_2x2x2(0, 2, 0))
    cols3.extend(schoolbook_block([0,2], [0,1], [2,3,4]))
    cols3.extend(schoolbook_block([0,2], [2,3], [2,3,4]))
    cols3.extend(schoolbook_block([1], [0,1,2,3], [0,1,2,3,4]))
    if len(cols3) < best_rank:
        best_rank = len(cols3)
        best_columns = cols3

    # Verify the best decomposition
    total_rank = best_rank
    U_final = np.zeros((n*m, total_rank), dtype=int)
    V_final = np.zeros((m*p, total_rank), dtype=int)
    W_final = np.zeros((p*n, total_rank), dtype=int)
    for c, (U_c, V_c, W_c) in enumerate(best_columns):
        U_final[:, c] = U_c
        V_final[:, c] = V_c
        W_final[:, c] = W_c
    
    # Verify
    np.random.seed(42)
    valid = True
    for _ in range(5):
        A_test = np.random.randint(-10, 10, (n, m))
        B_test = np.random.randint(-10, 10, (m, p))
        C_expected = A_test @ B_test
        C_actual = np.zeros((n, p), dtype=int)
        for c in range(total_rank):
            fa = np.dot(U_final[:, c], A_test.reshape(-1))
            fb = np.dot(V_final[:, c], B_test.reshape(-1))
            Wc_reshaped = W_final[:, c].reshape(p, n).T
            C_actual += Wc_reshaped * (fa * fb)
        if not np.allclose(C_actual, C_expected):
            valid = False
            break
    
    if not valid:
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=int)
        V = np.zeros((m*p, rank), dtype=int)
        W = np.zeros((p*n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
