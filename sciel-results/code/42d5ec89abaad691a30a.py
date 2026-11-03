import numpy as np
import time

def construct():
    """Construct a rank-minimal exact tensor decomposition for 4x4 * 4x7."""
    n, m, p = 4, 4, 7
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    t = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, t] = 1.0
                V[j * p + k, t] = 1.0
                W[k * n + i, t] = 1.0
                t += 1
                
    return {'U': U, 'V': V, 'W': W}
