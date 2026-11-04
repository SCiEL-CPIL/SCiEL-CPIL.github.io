import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # We'll use a direct rank-based approach.
    # For 3x5 * 5x7, the matrix multiplication tensor has rank at most 105 (schoolbook).
    # We can try to find a lower rank decomposition.
    
    # Let's use a known decomposition or construct one.
    # For now, let's use the schoolbook decomposition as a baseline and see if we can improve.
    
    # Schoolbook: rank = n*m*p = 3*5*7 = 105
    rank = 105
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For each (i, j, k), we have a term A[i,j] * B[j,k] contributing to C[i,k]
    # In the tensor decomposition, each rank-1 component is:
    # U[i*m + j, r] * V[j*p + k, r] * W[k*n + i, r] = delta[i,j,k]
    
    # Schoolbook: for each (i, j, k), set U[i*m+j, r] = 1, V[j*p+k, r] = 1, W[k*n+i, r] = 1
    # where r = i*m*p + j*p + k
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m * p + j * p + k
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
    
    return {"U": U, "V": V, "W": W}
