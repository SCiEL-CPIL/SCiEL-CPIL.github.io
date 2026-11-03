import numpy as np

def construct():
    n, m, p = 3, 5, 5
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, idx] = 1
                V[v_idx, idx] = 1
                W[w_idx, idx] = 1
                idx += 1
                
    return {'U': U, 'V': V, 'W': W}
