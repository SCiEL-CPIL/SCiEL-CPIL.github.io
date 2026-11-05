import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 3 * 7 = 21
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    for j in range(m):
        for k in range(p):
            r = j * p + k
            # U[i*m+j, r] = 1 for all i
            for i in range(n):
                U[i * m + j, r] = 1
            # V[j*p+k, r] = 1
            V[j * p + k, r] = 1
            # W[k*n+i, r] = 1 for all i
            for i in range(n):
                W[k * n + i, r] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
