import numpy as np

def construct():
    m = 4
    n = 4
    p = 9
    rank = m * n
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for i in range(m):
        for j in range(n):
            # U index: i * m + j ? 
            # Prompt: U shape (n*m, rank). 
            # Usually index is i*m + j or j*m + i.
            # Let's assume row index of U is i*m + j.
            u_idx = i * m + j
            U[u_idx, r] = 1.0
            
            # V index: j * p + k
            # V_{(j, k), r} = 1 for all k
            for k in range(p):
                v_idx = j * p + k
                V[v_idx, r] = 1.0
            
            # W index: k * n + i
            # W_{(k, i), r} = 1 for all k
            for k in range(p):
                w_idx = k * n + i
                W[w_idx, r] = 1.0
            
            r += 1
            
    return {'U': U, 'V': V, 'W': W}
