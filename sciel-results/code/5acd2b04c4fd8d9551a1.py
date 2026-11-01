"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition for 3x3 matrix multiplication.
    Currently uses the schoolbook decomposition (Rank 27).
    A Rank 23 decomposition would be ideal but requires exact coefficients.
    """
    n, m, p = 3, 3, 3
    rank = n * m * p  # 27
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, c] = 1
                # V[j*p + k, c] = 1
                # W[k*n + i, c] = 1
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
