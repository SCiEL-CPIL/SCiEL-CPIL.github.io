import numpy as np

def construct():
    n, m, p = 3, 4, 5
    base_rank = n * m * p  # 60
    
    # Schoolbook decomposition
    U = np.zeros((n * m, base_rank), dtype=int)
    V = np.zeros((m * p, base_rank), dtype=int)
    W = np.zeros((p * n, base_rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
