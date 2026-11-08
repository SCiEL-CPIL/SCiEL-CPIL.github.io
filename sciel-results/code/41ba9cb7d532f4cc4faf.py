import numpy as np

def construct():
    n = 2
    m = 4
    p = 6
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
                
    return {'U': U, 'V': V, 'W': W}
