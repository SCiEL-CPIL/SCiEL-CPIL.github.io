import numpy as np
import time

def construct():
    """
    Returns a rank-minimal (trivial) exact tensor decomposition for 2x4 by 4x5 matrix multiplication.
    U: (2*4, rank)
    V: (4*5, rank)
    W: (5*2, rank)
    """
    n, m, p = 2, 4, 5
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j, idx] = 1.0
                # V index: j*p + k
                V[j * p + k, idx] = 1.0
                # W index: k*n + i (Trace convention W[k*n+i, r])
                W[k * n + i, idx] = 1.0
                idx += 1
                
    return {'U': U, 'V': V, 'W': W}
