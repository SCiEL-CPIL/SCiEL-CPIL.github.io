import numpy as np
import time

n, m, p = 2, 5, 6

def construct():
    # The task is to decompose the tensor where T[i,j,k] = 1 for all i,j,k.
    # This is the all-ones tensor.
    # The rank of the all-ones tensor is 1.
    # U[i*m+j, 0] = 1
    # V[j*p+k, 0] = 1
    # W[k*n+i, 0] = 1
    # Sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 * 1 * 1 = 1.
    
    rank = 1
    U = np.ones((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
