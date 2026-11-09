# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    # Corrected Strassen terms for 2x2 matrix multiplication
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    #
    # C11 = M1 - M4 + M6 + M7
    # C12 = M2 + M3
    # C21 = M4 + M5
    # C22 = M1 - M2 + M3 + M5
    
    rank = 189
    U = np.zeros((n*m, rank), dtype=np.int64)
    V = np.zeros((m*p, rank), dtype=np.int64)
    W = np.zeros((p*n, rank), dtype=np.int64)
    
    # strassen_terms[t] = {'U': [(lr, lc, coeff), ...], 'V': [...], 'W': [...]}
    # U/V/W entries are (local_row, local_col, coeff)
    
    strassen_terms = [
        # M1
        {'U': [(0, 0, 1), (1, 1, 1)], 'V': [(0, 0, 1), (1, 1, 1)], 'W': [(0, 0, 1), (1, 1, 1)]},
        # M2
        {'U': [(1, 0, 1), (1, 1, 1)], 'V': [(0, 0, 1)], 'W': [(0, 1, 1), (1, 1, -1)]},
        # M3
        {'U': [(0, 0, 1)], 'V': [(0, 1, 1), (1, 1, -1)], 'W': [(0, 1, 1), (1, 1, 1)]},
        # M4
        {'U': [(1, 1, 1)], 'V': [(1, 0, 1), (0, 0, -1)], 'W': [(0, 0, -1), (1, 0, 1)]},
        # M5
        {'U': [(0, 0, 1), (0, 1, 1)], 'V': [(1, 1, 1)], 'W': [(1, 0, 1), (1, 1, 1)]},
        # M6
        {'U': [(1, 0, 1), (0, 0, -1)], 'V': [(0, 0, 1), (0, 1, 1)], 'W': [(0, 0, 1)]},
        # M7
        {'U': [(0, 1, 1), (1, 1, -1)], 'V': [(1, 0, 1), (1, 1, 1)], 'W': [(0, 0, 1)]}
    ]
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for t in range(7):
                    col_idx = (i * 9 + j * 3 + k) * 7 + t
                    term = strassen_terms[t]
                    
                    # U factors: A block (i, k)
                    for lr, lc, coeff in term['U']:
                        u_row = (2 * i + lr) * m + (2 * k + lc)
                        U[u_row, col_idx] += coeff
                        
                    # V factors: B block (k, j)
                    for lr, lc, coeff in term['V']:
                        v_row = (2 * k + lr) * p + (2 * j + lc)
                        V[v_row, col_idx] += coeff
                        
                    # W factors: C block (i, j)
                    # W index: k*n + i -> j*n + i? 
                    # C is p x n. Index (j, i) in C corresponds to row j, col i.
                    # W shape (p*n, rank). Row index = j * n + i.
                    # Local (lr, lc) in 2x2 block C[i, j].
                    # Global row in C = 2*i + lr. Global col in C = 2*j + lc.
                    # W row index = (2*j + lc) * n + (2*i + lr).
                    for lr, lc, coeff in term['W']:
                        w_row = (2 * j + lc) * n + (2 * i + lr)
                        W[w_row, col_idx] += coeff

    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
