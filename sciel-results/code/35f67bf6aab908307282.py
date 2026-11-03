import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin
    
    n, m, p = 2, 5, 5
    # Schoolbook rank is n * m * p
    rank = n * m * p
    
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Index mapping:
    # A is n x m. Index (i, j) -> i*m + j.
    # B is m x p. Index (j, k) -> j*p + k.
    # C is n x p. Index (i, k) -> i*p + k.
    # But W is defined as W[k*n + i, r] in the prompt?
    # "output factor follows the official trace convention W[k*n+i,r]"
    # This implies W row index is k*n + i.
    # Standard C index is i*p + k.
    # Let's check the prompt: "W[k*n+i,r]".
    # So W row index corresponds to (k, i) where k is col, i is row.
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Term for A[i,j] * B[j,k] contributing to C[i,k]
                # U index: i*m + j
                U[i * m + j, r] = 1.0
                # V index: j*p + k
                V[j * p + k, r] = 1.0
                # W index: k*n + i
                W[k * n + i, r] = 1.0
                r += 1
                
    return {'U': U, 'V': V, 'W': W}
