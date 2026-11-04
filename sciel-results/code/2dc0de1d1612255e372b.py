import numpy as np

def construct():
    n, m, p = 2, 4, 8
    rank = 32
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for i in range(n):
        for a in range(m):
            for c in range(p):
                r = i * m * p + a * p + c
                U[i*m + a, r] = 1
                V[a*p + c, r] = 1
                W[c*n + i, r] = 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
