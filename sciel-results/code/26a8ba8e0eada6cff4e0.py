import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 105
    num_u = n * m  # 15
    num_v = m * p  # 35
    num_w = p * n  # 21
    
    U = np.zeros((num_u, rank), dtype=int)
    V = np.zeros((num_v, rank), dtype=int)
    W = np.zeros((num_w, rank), dtype=int)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, r] = 1
                V[v_idx, r] = 1
                W[w_idx, r] = 1
                r += 1
    
    return {"U": U, "V": V, "W": W}
