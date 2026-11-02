"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 * 3x8 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS
    # Use Schoolbook decomposition to guarantee validity
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill U, V, W for schoolbook
    # Index mapping: r = i*m*p + j*p + k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
