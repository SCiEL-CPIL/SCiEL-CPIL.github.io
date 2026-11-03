import numpy as np
import time

n, m, p = 3, 5, 7
R = 105  # rank

def construct():
    """
    Returns a rank-105 exact tensor decomposition of matrix multiplication 3x5 by 5x7.
    This is the trivial Kronecker product decomposition.
    """
    # U: (n*m, R)
    # V: (m*p, R)
    # W: (p*n, R)
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, r] = 1.0
                V[v_idx, r] = 1.0
                W[w_idx, r] = 1.0
                
                r += 1
                
    return {'U': U, 'V': V, 'W': W}
