import numpy as np

def construct():
    n, m, p = 2, 4, 6
    rank = n * m * p
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Initialize matrices with zeros
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Schoolbook decomposition
    # For each (i, j, k), we have a term U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # We assign a unique index r for each (i, j, k)
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, r] = 1
                V[j * p + k, r] = 1
                W[k * n + i, r] = 1
                r += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
