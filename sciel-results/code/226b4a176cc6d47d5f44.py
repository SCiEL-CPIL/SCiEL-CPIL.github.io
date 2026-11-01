import numpy as np

def construct():
    n, m, p = 3, 3, 3
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i, j -> i*m + j
                # V index: j, k -> j*p + k
                # W index: k, i -> k*n + i (Task: W[k*n+i, r])
                
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, idx] = 1.0
                V[v_idx, idx] = 1.0
                W[w_idx, idx] = 1.0
                idx += 1
                
    return {'U': U, 'V': V, 'W': W}
