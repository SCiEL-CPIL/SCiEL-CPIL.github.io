import numpy as np

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    total_rank = n * m * p
    
    U_mat = np.zeros((n * m, total_rank), dtype=float)
    V_mat = np.zeros((m * p, total_rank), dtype=float)
    W_mat = np.zeros((p * n, total_rank), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U_mat[u_idx, idx] = 1.0
                V_mat[v_idx, idx] = 1.0
                W_mat[w_idx, idx] = 1.0
                idx += 1
                
    return {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
