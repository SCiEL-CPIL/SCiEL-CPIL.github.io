"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=6
    
    # Schoolbook decomposition: rank = n*m*p = 48
    # For each (i,j,k), we have a rank-1 term:
    #   U[i*m+j, col] = 1
    #   V[j*p+k, col] = 1
    #   W[k*n+i, col] = 1
    # This exactly decomposes the matrix multiplication tensor.
    
    rank = n * m * p
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
