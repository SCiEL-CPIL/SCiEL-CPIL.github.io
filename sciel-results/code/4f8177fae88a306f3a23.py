import numpy as np

def construct():
    n, m, p = 2, 5, 5
    rank = n * m * p  # 50
    
    # Build factor matrices using structured indexing
    # U has shape (n*m, rank) = (10, 50)
    # V has shape (m*p, rank) = (25, 50)  
    # W has shape (p*n, rank) = (10, 50)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # For each rank index r, decompose into (i, j, k)
    # U[i*m + j, r] = 1
    # V[j*p + k, r] = 1
    # W[k*n + i, r] = 1
    # This ensures: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # = sum_j A[i,j] * B[j,k] for each (i,k)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * (m * p) + j * p + k
                U[i * m + j, r] = 1.0
                V[j * p + k, r] = 1.0
                W[k * n + i, r] = 1.0
    
    # Convert to lists for the required return format
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
