import time
import numpy as np

def construct():
    n, m, p = 2, 2, 13
    nm = n * m
    mp = m * p
    pn = p * n
    
    R = n * m * p
    U = np.zeros((nm, R), dtype=float)
    V = np.zeros((mp, R), dtype=float)
    W = np.zeros((pn, R), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                
                U[a_idx, col] = 1.0
                V[b_idx, col] = 1.0
                W[c_idx, col] = 1.0
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
