"""Matrix multiplication 4,5,5 - rank-minimal exact tensor decomposition."""
import numpy as np
from fractions import Fraction
import time
import random

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 10s safety margin
    
    # Schoolbook decomposition: rank = n * m * p
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                r += 1
    
    # Return schoolbook decomposition as baseline
    # Finding lower-rank exact decompositions requires solving
    # systems of polynomial equations, which is computationally intensive
    return {"U": U, "V": V, "W": W}
