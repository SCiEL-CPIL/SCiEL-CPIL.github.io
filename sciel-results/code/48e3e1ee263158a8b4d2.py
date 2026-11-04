"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [3, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # The most robust and fast solution is the schoolbook algorithm.
    # Rank = n * m * p = 3 * 3 * 4 = 36.
    # This construction is exact, uses integers, and takes O(1) time (negligible).
    # It avoids the timeout issues of the previous gradient-based search.
    
    rank = n * m * p
    
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index mapping:
                # U index: i * m + j
                # V index: j * p + k
                # W index: k * n + i (following W[k*n+i, r] convention)
                
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
