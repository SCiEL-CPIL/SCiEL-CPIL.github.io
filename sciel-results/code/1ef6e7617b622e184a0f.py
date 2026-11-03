"""Rectangular Strassen approach for 3x3x8 matrix multiplication."""
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 3x3x8 multiplication using the naive rank-72 decomposition.
    
    The naive decomposition has rank = n*m*p = 3*3*8 = 72.
    Each term corresponds to one entry of A, one entry of B, and one entry of C.
    
    Indexing convention:
    - U has shape (n*m, rank) = (9, 72), indexed by (i,j) -> i*m+j
    - V has shape (m*p, rank) = (24, 72), indexed by (j,k) -> j*p+k  
    - W has shape (p*n, rank) = (24, 72), indexed by (k,i) -> k*n+i (trace convention)
    
    For each (i,j,k) triple, we have:
    - U[i*m+j, r] = 1
    - V[j*p+k, r] = 1
    - W[k*n+i, r] = 1
    where r is the column index for that term.
    """
    n, m, p = 3, 3, 8
    rank = n * m * p  # 72
    
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
