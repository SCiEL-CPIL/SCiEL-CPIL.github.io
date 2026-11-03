import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Constructs the schoolbook tensor decomposition for multiplying a 3x3 matrix by a 3x5 matrix.
    
    The multiplication tensor T[i,j,k] corresponds to the element at row i, column k of the result,
    which is sum_j A[i,j] * B[j,k].
    
    We decompose T into rank-1 terms. The schoolbook decomposition has one term for each (i, j, k) triple.
    Total rank = n * m * p = 3 * 3 * 5 = 45.
    
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    
    Indexing conventions:
    U[i*m + j, c]
    V[j*p + k, c]
    W[k*n + i, c]
    """
    n, m, p = DIMENSIONS  # n=3, m=3, p=5
    rank = n * m * p
    
    # Initialize matrices with zeros
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
