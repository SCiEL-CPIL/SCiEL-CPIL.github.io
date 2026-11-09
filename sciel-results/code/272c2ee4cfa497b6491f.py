import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, np_ = n*m, m*p, p*n
    
    # The schoolbook rank is n * m * p
    rank = n * m * p
    
    # Initialize factor matrices with zeros
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((np_, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index mapping:
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W]
    }
