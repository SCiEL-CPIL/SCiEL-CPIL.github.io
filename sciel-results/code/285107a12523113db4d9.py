"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import itertools

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    # We need to find a rank-minimal exact tensor decomposition for multiplying
    # a 4x4 matrix by a 4x9 matrix.
    # The output factor W follows the convention W[k*n+i, r].
    # U has shape (n*m, rank) = (16, rank)
    # V has shape (m*p, rank) = (36, rank)
    # W has shape (p*n, rank) = (36, rank)
    
    # The tensor T[i, j, k] = 1 for all i in 0..n-1, j in 0..m-1, k in 0..p-1
    # is decomposed as sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For the standard schoolbook algorithm, rank = n*m*p = 4*4*9 = 144.
    # We can potentially do better using Strassen-like algorithms or other optimizations.
    
    # Let's start with the schoolbook decomposition and see if we can improve it.
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
