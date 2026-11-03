"""Program for finding a rank-minimal exact tensor decomposition for 2x4*4x5 matrix multiplication."""
import numpy as np
import random
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-40 tensor decomposition for 2x4 * 4x5 matrix multiplication.
    
    The multiplication tensor T has indices (i, j, k) where i in [0,1], j in [0,3], k in [0,4].
    T[(i,j), (j,k)] = 1 if the indices match (standard multiplication tensor).
    The output factor follows the official trace convention W[k*n+i,r].
    """
    n, m, p = DIMENSIONS  # 2, 4, 5
    rank = n * m * p  # 40 for schoolbook
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
