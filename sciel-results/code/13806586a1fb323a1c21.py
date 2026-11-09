"""Matrix multiplication 2x3 * 3x6: rank-minimal tensor decomposition."""
import time
import sys
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x6 matrix multiplication.
    
    Convention: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    with W[k*n+i, t] indexing.
    """
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    # Schoolbook rank = 36
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
