"""
Structured sparse decomposition for 4x5x5 matrix multiplication.
Uses block-diagonal with shift pattern to construct a rank-100 solution.
Interface: construct() -> {'U', 'V', 'W'}
"""
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 100
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
                r += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
