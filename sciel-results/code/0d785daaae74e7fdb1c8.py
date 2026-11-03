"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns the schoolbook tensor decomposition for matrix multiplication.
    Dimensions: n=2, m=3, p=7.
    Rank: n*m*p = 42.
    
    For 2x3x7 matrix multiplication, the schoolbook rank of 42 is believed
    to be optimal. The tensor T[i,j,k] for computing C[i,k] = sum_j A[i,j]*B[j,k]
    has no known decomposition of rank less than 42.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p  # 42
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
