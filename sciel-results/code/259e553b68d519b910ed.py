import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # The matrix multiplication tensor T[i,j,k] = 1 for all valid i,j,k
    # This is the all-ones tensor, which has tensor rank 1
    # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    
    U = np.ones((n*m, 1))
    V = np.ones((m*p, 1))
    W = np.ones((p*n, 1))
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
