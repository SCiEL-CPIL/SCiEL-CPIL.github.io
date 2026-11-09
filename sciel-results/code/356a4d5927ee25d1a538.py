"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # safety margin before 150s kill
    
    # Schoolbook decomposition
    schoolbook_rank = n * m * p
    U = [[0] * schoolbook_rank for _ in range(n * m)]
    V = [[0] * schoolbook_rank for _ in range(m * p)]
    W = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    # Try to find a better decomposition using a known rank-23 3x3x3 decomposition
    # We construct a rank-46 decomposition for 3x3x6 by using two independent rank-23 3x3x3 decompositions
    # This is valid because 3x3x6 is the direct sum of two 3x3x3 tensors.
    
    # Hardcode a rank-23 decomposition for 3x3x3
    # If we don't have the exact coefficients, we fall back to schoolbook.
    # For now, we return schoolbook to ensure validity and speed.
    # A future improvement would be to hardcode the Lader rank-23 coefficients.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
