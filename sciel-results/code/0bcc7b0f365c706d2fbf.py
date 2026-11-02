import numpy as np

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=3, p=5
    # Rank 15 decomposition
    rank = m * p  # 3 * 5 = 15
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Indices:
    # U: (i*m + j, t)
    # V: (j*p + k, t)
    # W: (k*n + i, t)
    # t corresponds to (j, k) pair
    
    t = 0
    for j in range(m):
        for k in range(p):
            # Term t uses A[:, j] and B[j, :]
            # U part: A[i, j] for i=0,1
            U[0 * m + j, t] = 1
            U[1 * m + j, t] = 1
            
            # V part: B[j, k]
            V[j * p + k, t] = 1
            
            # W part: C[i, k] for i=0,1
            W[k * n + 0, t] = 1
            W[k * n + 1, t] = 1
            
            t += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
