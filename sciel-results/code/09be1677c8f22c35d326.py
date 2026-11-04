"""Program for rank-minimal exact tensor decomposition of 2x3 * 3x4 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns an exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
    Returns the schoolbook decomposition (rank 24) to ensure validity and avoid numerical errors.
    """
    n, m, p = DIMENSIONS  # 2, 3, 4
    dim_U = n * m  # 6
    dim_V = m * p  # 12
    dim_W = p * n  # 8
    rank_school = n * m * p  # 24
    
    U = [[0] * rank_school for _ in range(dim_U)]
    V = [[0] * rank_school for _ in range(dim_V)]
    W = [[0] * rank_school for _ in range(dim_W)]
    
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
