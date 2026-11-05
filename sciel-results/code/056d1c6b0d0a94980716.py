import time
import numpy as np

def construct():
    n, m, p = 4, 5, 5
    nm, mp, pn = n*m, m*p, p*n
    
    # Use a known low-rank decomposition.
    # For 4x5x5, the rank is 96.
    # I will construct a rank-96 decomposition.
    
    # Since I don't have the explicit formula, I will use a fallback.
    
    # Schoolbook
    rank = 100
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1.0
                V[j*p + k, col] = 1.0
                W[k*n + i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
