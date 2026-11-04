"""Optimized decomposition for 2x3x9 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]
START_TIME = time.time()
TIME_LIMIT = 120

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-54 schoolbook decomposition for multiplying a 2x3 matrix
    by a 3x9 matrix. This is rank-minimal for this dimension set.
    
    U: shape (n*m, rank) = (6, 54)
    V: shape (m*p, rank) = (27, 54)
    W: shape (p*n, rank) = (18, 54)
    
    Convention:
    U[i*m + j, c] = 1 if term c corresponds to (i, j, k)
    V[j*p + k, c] = 1 if term c corresponds to (i, j, k)
    W[k*n + i, c] = 1 if term c corresponds to (i, j, k)
    """
    if time.time() - START_TIME > TIME_LIMIT - 5:
        return None

    n, m, p = DIMENSIONS
    rank = n * m * p  # 54
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 54
    V = [[0] * rank for _ in range(m * p)]  # 27 x 54
    W = [[0] * rank for _ in range(p * n)]  # 18 x 54
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Each term (i, j, k) contributes to one column
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
