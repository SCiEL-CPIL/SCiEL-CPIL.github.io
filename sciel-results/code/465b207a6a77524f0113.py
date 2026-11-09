import numpy as np

def construct():
    n, m, p = 2, 4, 5
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Map indices to rank
                r = i * (m * p) + j * p + k
                
                # U index: (i, j) -> i*m + j
                u_idx = i * m + j
                # V index: (j, k) -> j*p + k
                v_idx = j * p + k
                # W index: (k, i) -> k*n + i
                w_idx = k * n + i
                
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
                
    return {'U': U, 'V': V, 'W': W}
