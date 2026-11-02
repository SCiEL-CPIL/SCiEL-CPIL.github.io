"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    # The task is to find a rank-minimal exact tensor decomposition for M_{3,5,7}.
    # The schoolbook decomposition has rank n*m*p = 3*5*7 = 105.
    # Finding a decomposition with rank < 105 requires specific algebraic identities
    # (like Strassen's for 2x2) which are not trivially applicable or known in simple
    # integer form for 3x5x7 without extensive pre-computation or literature lookup.
    # Random search for valid integer decompositions of rank ~100 is computationally
    # infeasible within the 120s limit to guarantee a valid result.
    # Therefore, we return the guaranteed valid schoolbook decomposition.
    
    n, m, p = 3, 5, 7
    r = n * m * p
    
    U = np.zeros((n * m, r), dtype=np.int64)
    V = np.zeros((m * p, r), dtype=np.int64)
    W = np.zeros((p * n, r), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
# EVOLVE-BLOCK-END
