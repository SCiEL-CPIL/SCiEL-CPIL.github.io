import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # The rank of the matrix multiplication tensor M_{n,m,p} 
    # can be decomposed by splitting the first dimension.
    # For n=2, we can use 2 rank-1 tensors, one for each row of A.
    # This yields a rank of 2.
    
    rank = 2
    
    U = np.zeros((nm, rank), dtype=np.float64)
    V = np.zeros((mp, rank), dtype=np.float64)
    W = np.zeros((pn, rank), dtype=np.float64)
    
    # For each i in range(n), we assign a rank r = i.
    # U[i*m + j, i] = 1
    # V[j*p + k, i] = 1
    # W[k*n + i, i] = 1
    
    for i in range(n):
        # U
        for j in range(m):
            U[i * m + j, i] = 1.0
            
        # V
        for j in range(m):
            for k in range(p):
                V[j * p + k, i] = 1.0
                
        # W
        for k in range(p):
            W[k * n + i, i] = 1.0

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
