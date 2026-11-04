import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank-m decomposition:
    # U[i*m+j, j'] = delta(j, j')
    # V[j*p+k, j'] = 1
    # W[k*n+i, j'] = 1
    # Then sum_{j'} U[i*m+j,j'] * V[j*p+k,j'] * W[k*n+i,j'] = sum_{j'} delta(j,j') * 1 * 1 = 1
    rank = m
    U = np.zeros((n * m, rank))
    V = np.ones((m * p, rank))
    W = np.ones((p * n, rank))
    
    for i in range(n):
        for j in range(m):
            U[i * m + j, j] = 1
    
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
