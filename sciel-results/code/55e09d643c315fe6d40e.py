import numpy as np
import time

DIMENSIONS = [2, 4, 6]
n, m, p = DIMENSIONS

def construct():
    """
    Returns the schoolbook tensor decomposition for matrix multiplication
    of size (n, m) x (m, p).
    
    The rank is n * m * p.
    U has shape (n*m, rank)
    V has shape (m*p, rank)
    W has shape (p*n, rank)
    
    The decomposition satisfies:
    C[i, k] = sum_r W[k*n + i, r] * (sum_j U[i*m + j, r] * A[i, j]) * (sum_j V[j*p + k, r] * B[j, k])
    """
    rank = n * m * p
    
    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # For each (i, j, k) term in the sum, we create a rank-1 slice
                # U selects A[i, j]
                U[i * m + j][column] = 1
                # V selects B[j, k]
                V[j * p + k][column] = 1
                # W places the result into C[i, k]
                # Note the trace convention: W[k*n + i, r]
                W[k * n + i][column] = 1
                
                column += 1
                
    return {"U": U, "V": V, "W": W}
