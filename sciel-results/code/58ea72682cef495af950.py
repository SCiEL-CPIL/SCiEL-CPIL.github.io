"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank-32 schoolbook decomposition of 2x4x4 matrix multiplication.
    # U: (n*m, rank) = (8, 32)
    # V: (m*p, rank) = (16, 32)
    # W: (p*n, rank) = (8, 32)
    #
    # Tensor T[i, j, k] = 1 where:
    #   i = row of A (0..1)
    #   j = col of A / row of B (0..3)
    #   k = col of B / col of C (0..3)
    #
    # Output index in W: k*n + i
    # U index: i*m + j
    # V index: j*p + k
    
    n, m, p = DIMENSIONS
    rank = n * m * p  # 32
    
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
