import time
import numpy as np
import itertools

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Trivial structured decomposition
    # Rank R = n * m
    # U is (n*m) x R
    # V is (m*p) x R
    # W is (p*n) x R
    
    R = n * m
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    for r in range(R):
        i = r // m
        j = r % m
        
        # U[i*m+j, r] = 1
        U[i * m + j][r] = 1
        
        # V[j*p+k, r] = 1 for all k
        for k in range(p):
            V[j * p + k][r] = 1
            
        # W[k*n+i, r] = 1 for all k
        for k in range(p):
            W[k * n + i][r] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
