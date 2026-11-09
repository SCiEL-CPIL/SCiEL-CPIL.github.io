import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    """Constructs the factor matrices for 5x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    
    # The schoolbook algorithm for matrix multiplication has rank n*m*p
    # U, V, W have shapes (n*m, R), (m*p, R), (p*n, R)
    R = n * m * p
    
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i (Trace convention W[k*n+i, r])
                
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Return the factors as lists of lists as per interface
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
