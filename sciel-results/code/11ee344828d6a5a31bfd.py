<<<<<<< SEARCH
U[i * m + j][column] = 1
V[j * p + k][column] = 1
W[k * n + i][column] = 1
=======
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = 3, 4, 6
    
    # The trivial tensor decomposition has rank n*m*p = 3*4*6 = 72
    # U shape: (n*m, rank) = (12, 72)
    # V shape: (m*p, rank) = (24, 72)
    # W shape: (p*n, rank) = (18, 72)
    
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Construct the trivial decomposition
    # For each (i, j, k) in range(n) x range(m) x range(p):
    # U[i*m + j, idx] = 1
    # V[j*p + k, idx] = 1
    # W[k*n + i, idx] = 1
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    
    # Verify correctness (optional, but good for safety)
    # The tensor T[i, j, k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For the trivial decomposition, T[i,j,k] = 1 for all i,j,k, which matches matrix multiplication.
    
    return {'U': U, 'V': V, 'W': W}
>>>>>>> REPLACE
