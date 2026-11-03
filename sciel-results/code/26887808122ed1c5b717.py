"""
Optimized program for matrix multiplication tensor decomposition.
Dimensions: 3x3 * 3x5.
Known optimal rank: 45 (schoolbook).
"""

import numpy as np
import time

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the factor matrices U, V, W for the matrix multiplication tensor
    of dimensions 3x3 * 3x5.
    
    The decomposition is of rank 45, which is the schoolbook rank.
    U has shape (n*m, rank) = (9, 45)
    V has shape (m*p, rank) = (15, 45)
    W has shape (p*n, rank) = (15, 45)
    
    The entries are 0 or 1.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (following the trace convention W[k*n+i, r])
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
