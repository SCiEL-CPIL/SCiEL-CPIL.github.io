"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    """
    Exact tensor decomposition for 4x4 * 4x7 matrix multiplication.
    Uses the standard (schoolbook) algorithm which has rank n*m*p = 4*4*7 = 112.
    This guarantees finite, integer coefficients and correctness.
    """
    n, m, p = DIMENSIONS  # 4, 4, 7
    
    rank = n * m * p  # 112
    
    # Initialize matrices with zeros
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j][r] = 1.0
                # V index: j*p + k
                V[j * p + k][r] = 1.0
                # W index: k*n + i (per task convention W[k*n+i, r])
                W[k * n + i][r] = 1.0
                r += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
