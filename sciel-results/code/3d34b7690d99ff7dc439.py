import numpy as np

def construct():
    n = 4
    m = 4
    p = 6
    
    # Trivial decomposition rank
    rank = n * m * p
    # 4 * 4 * 6 = 96
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: (i, j) -> i*m + j
                u_idx = i * m + j
                # V index: (j, k) -> j*p + k
                v_idx = j * p + k
                # W index: (i, k) -> i*p + k
                w_idx = i * p + k
                
                U[u_idx, idx] = 1.0
                V[v_idx, idx] = 1.0
                W[w_idx, idx] = 1.0
                idx += 1
                
    return {'U': U, 'V': V, 'W': W}
