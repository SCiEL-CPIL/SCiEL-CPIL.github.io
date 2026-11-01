import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Use the standard schoolbook decomposition.
    # Rank R = n * m * p = 3 * 3 * 6 = 54.
    R = n * m * p
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                
                U[a, r] = 1.0
                V[b, r] = 1.0
                W[c, r] = 1.0
                r += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
