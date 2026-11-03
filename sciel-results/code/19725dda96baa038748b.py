import numpy as np
import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    
    # The target tensor is the all-ones tensor of size n x m x p.
    # The condition is sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 for all i,j,k.
    # A rank-1 decomposition exists:
    # U[i,j,0] = 1
    # V[j,k,0] = 1
    # W[k,i,0] = 1
    # Sum = 1 * 1 * 1 = 1.
    
    rank = 1
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
