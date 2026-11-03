"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    
    # Use a structured approach with Z_4 symmetry to find a lower-rank decomposition
    # We'll exploit the cyclic structure in the middle dimension (m=4)
    
    # Strategy: Group the 60 schoolbook terms by j mod 4 and try to find
    # linear combinations that reduce the total count
    
    # The key insight: for fixed i and k, we have terms for j=0,1,2,3
    # T[i*m+j, j*p+k, k*n+i] = 1 for each j
    
    # We want to see if we can replace some of these with shared factors
    
    # Let's try a more direct approach: use the known result that
    # matrix multiplication tensors can sometimes be decomposed with rank < nmp
    
    # For 3x4x5, let's attempt to find a rank-55 decomposition
    
    # First, let's try a simple optimization: if there are any linear dependencies
    # among the standard basis tensors, we can exploit them
    
    # Actually, let's just return the schoolbook decomposition for now,
    # as finding a lower rank decomposition is non-trivial
    
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
