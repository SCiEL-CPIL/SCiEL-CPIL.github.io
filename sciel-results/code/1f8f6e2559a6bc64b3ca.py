"""Optimized program for 4x5*5x5 matrix multiplication tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook decomposition of rank 100 for 4x5 * 5x5 matrix multiplication.
    
    The tensor T[i,j,k] = delta_{i,j} delta_{j,k} (in appropriate indexing)
    is decomposed as sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c].
    
    Schoolbook: one term per (i,j,k) triple, giving rank = n*m*p = 100.
    
    The trace convention for W is W[k*n + i, r] as specified.
    """
    n, m, p = DIMENSIONS  # 4, 5, 5
    rank = n * m * p  # 100
    
    # Build factor matrices as lists of lists with integer entries
    U = [[0] * rank for _ in range(n * m)]  # shape (20, 100)
    V = [[0] * rank for _ in range(m * p)]  # shape (25, 100)
    W = [[0] * rank for _ in range(p * n)]  # shape (20, 100)
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
