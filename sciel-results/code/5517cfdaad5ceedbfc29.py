import numpy as np

def construct():
    n, m, p = 3, 4, 4
    
    # Use a verified rank-48 trivial decomposition
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    idx = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a = i * m + k
                b = k * p + j
                c = j * n + i
                U[a, idx] = 1
                V[b, idx] = 1
                W[c, idx] = 1
                idx += 1
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
