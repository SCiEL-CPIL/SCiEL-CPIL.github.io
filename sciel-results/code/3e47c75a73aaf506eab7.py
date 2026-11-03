import numpy as np

def construct():
    n, m, p = 2, 4, 7
    rank = n * m * p
    
    # Initialize with zeros
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Construct decomposition
    # Each rank-1 term corresponds to a triplet (i, j, k)
    # i in 0..n-1, j in 0..m-1, k in 0..p-1
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                U[i * m + j, col] = 1
                # V index: j*p + k
                V[j * p + k, col] = 1
                # W index: k*n + i
                W[k * n + i, col] = 1
                col += 1
                
    # Convert to lists for the return interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
