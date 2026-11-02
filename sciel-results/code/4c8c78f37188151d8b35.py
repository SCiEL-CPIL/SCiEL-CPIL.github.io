"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

import time
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    """
    Return the schoolbook exact tensor decomposition for 2x3 * 3x9 matrix multiplication.
    
    The tensor T[i,j,k] = A[i,j] * B[j,k] has a trivial rank-nmp decomposition
    with one term per (i,j,k) triple. For n=2, m=3, p=9, rank = 54.
    
    U has shape (n*m, rank): U[i*m + j, c]
    V has shape (m*p, rank): V[j*p + k, c]  
    W has shape (p*n, rank): W[k*n + i, c]
    
    For each column c corresponding to (i,j,k):
      U[i*m+j, c] = 1
      V[j*p+k, c] = 1
      W[k*n+i, c] = 1
    
    This gives sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 if (i,j,k) matches, else 0,
    which is exactly the matrix multiplication tensor under the trace convention.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 54
    
    # Initialize with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
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
