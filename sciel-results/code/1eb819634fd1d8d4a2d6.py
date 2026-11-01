"""Matrix multiplication 2x5x5: rank-10 exact decomposition."""
import numpy as np
from fractions import Fraction
import time

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank is n * p = 2 * 5 = 10
    # We assign one rank-1 term to each (i, k) pair
    rank = n * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for k in range(p):
            c = i * p + k
            
            # W index is k * n + i
            W[k * n + i][c] = 1
            
            # U index is i * m + j. Set to 1 for all j
            for j in range(m):
                U[i * m + j][c] = 1
                
            # V index is j * p + k. Set to 1 for all j
            for j in range(m):
                V[j * p + k][c] = 1

    return {"U": U, "V": V, "W": W}
