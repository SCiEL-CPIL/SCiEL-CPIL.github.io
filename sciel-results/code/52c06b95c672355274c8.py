import numpy as np

def construct():
    n = 5
    rank = n * n * n  # 125
    
    U = np.zeros((n * n, rank))
    V = np.zeros((n * n, rank))
    W = np.zeros((n * n, rank))
    
    for r in range(rank):
        i = r // (n * n)
        k = (r // n) % n
        j = r % n
        
        u_idx = i * n + k
        v_idx = k * n + j
        w_idx = i * n + j
        
        U[u_idx, r] = 1
        V[v_idx, r] = 1
        W[w_idx, r] = 1
        
    return {'U': U, 'V': V, 'W': W}
