"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction
import itertools
import sys
import time

def construct():
    n, m, p = DIMENSIONS
    
    # The tensor T[i, j, k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # We seek rank r such that T = sum_{c=0}^{r-1} U[:,c] outer V[:,c] outer W[:,c]
    # where U is (n*m) x r, V is (m*p) x r, W is (p*n) x r
    # T[i*m+j, j*p+k, k*n+i] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Lower bound: rank >= max(n*m, m*p, p*n) = max(15, 35, 21) = 35
    # Schoolbook: rank = n*m*p = 105
    
    # Let's try to find a decomposition with rank < 105.
    # We'll use a heuristic approach: try different ranks and see if we can find a solution.
    
    # For now, let's return the schoolbook decomposition as a baseline.
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
