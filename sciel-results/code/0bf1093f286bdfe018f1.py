import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Schoolbook decomposition
    # T[i, j, k] = 1 if we are multiplying A[i,j] * B[j,k] to get C[i,k]
    # U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
    # For each (i, j, k), we have a term.
    # Total terms = n * m * p
    
    rank = n * m * p
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    # Verification (Optional, but good for debugging in dev, skip for speed in eval if confident)
    # The schoolbook decomposition is mathematically exact.
    
    # Convert to list of lists
    def to_list(mat):
        return [[float(mat[i][j]) for j in range(mat.shape[1])] for i in range(mat.shape[0])]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
