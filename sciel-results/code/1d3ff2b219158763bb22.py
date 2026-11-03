import numpy as np
import time

def construct():
    n, m, p = 3, 4, 6
    rank = n * m * p
    # Initialize U, V, W with zeros
    # U shape: (n*m, rank)
    # V shape: (m*p, rank)
    # W shape: (p*n, rank) -- Note: Task says W[k*n+i, r] which implies row index is k*n+i.
    # Standard W index is (i, k). Here row index is k*n + i.
    
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                u_idx = i * m + j
                # V index: j*p + k
                v_idx = j * p + k
                # W index: k*n + i (as per task description W[k*n+i,r])
                w_idx = k * n + i
                
                U[u_idx, r] = 1
                V[v_idx, r] = 1
                W[w_idx, r] = 1
                r += 1
                
    return {'U': U, 'V': V, 'W': W}
