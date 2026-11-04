"""Matrix multiplication 4x4x5 - rank-80 schoolbook decomposition."""
import numpy as np

def construct():
    """
    Construct a tensor decomposition for 4x4x5 matrix multiplication.
    Returns {'U': (16, rank), 'V': (20, rank), 'W': (20, rank)}
    where U[i*4+j, r], V[j*5+k, r], W[k*4+i, r]
    """
    n, m, p = 4, 4, 5
    
    # Schoolbook decomposition: rank = n * m * p = 4 * 4 * 5 = 80
    rank = n * m * p
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i * m + j
                U[i * m + j, r] = 1.0
                # V index: j * p + k
                V[j * p + k, r] = 1.0
                # W index: k * n + i
                W[k * n + i, r] = 1.0
                r += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
