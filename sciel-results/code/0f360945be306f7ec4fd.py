"""Matrix multiplication 3x5x5 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]

def construct():
    """Construct the factor matrices for 3x5x5 matrix multiplication.
    
    Returns dict with keys 'U', 'V', 'W' of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    Uses schoolbook decomposition (rank 75) as a safe baseline.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Schoolbook: for each (i,j,k), U[i*m+j, col]=1, V[j*p+k, col]=1, W[k*n+i, col]=1
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
