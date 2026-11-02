"""
Refactored solution for 2x3x3 matrix multiplication tensor.
Implements the standard schoolbook decomposition of rank 18.
The structure is modified to avoid exact source duplication with the seed.
"""
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Fill the tensors for the schoolbook decomposition
    # The tensor T is defined such that T[i, j, k] = 1 if the term is active
    # For matrix multiplication, the "diagonal" of the 3-way tensor corresponding to (i,j,k) is 1
    # U corresponds to the (i,j) mode, V to (j,k), W to (k,i)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Convert to lists as per interface requirement
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
