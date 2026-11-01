import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition has rank n*m*p = 27
    # This is a valid exact decomposition.
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # Indexing:
    # U index: i*m + j
    # V index: j*p + k
    # W index: k*n + i
    # T[u, v, w] = 1 if u=i*m+j, v=j*p+k, w=k*n+i else 0
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                U[u, col] = 1.0
                V[v, col] = 1.0
                W[w, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
