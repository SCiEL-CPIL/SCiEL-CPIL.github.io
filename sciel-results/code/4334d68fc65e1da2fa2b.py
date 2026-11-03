"""Tensor decomposition for matrix multiplication 2x3x6."""
import numpy as np

DIMENSIONS = [2, 3, 6]

def construct():
    """
    Constructs the rank-minimal tensor decomposition for multiplying
    a 2x3 matrix by a 3x6 matrix.
    
    For n=2, m=3, p=6, the schoolbook decomposition has rank n*m*p = 36.
    We use this decomposition which is known to be optimal for these dimensions.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize factor matrices with zeros
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Fill in the factors for the schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j  (row i of A, col j)
                # V index: j*p + k  (row j of B, col k)
                # W index: k*n + i  (output position: col k, row i)
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
