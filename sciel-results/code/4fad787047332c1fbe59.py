import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook algorithm for n x m * m x p
    # Rank = n * m * p
    rank = n * m * p
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float) # Note: W index is k*n + i? 
    # Let's verify W index convention from task: "W[k*n+i,r]"
    # If C is n x p, indices (i, j). i in [0,n), j in [0,p).
    # If index is k*n + i, then k is j (column) and i is i (row).
    # So W row index = j * n + i.
    
    r = 0
    for i in range(n):
        for k in range(m):
            for j in range(p):
                # A[i, k] * B[k, j] contributes to C[i, j]
                # U row index: i * m + k
                # V row index: k * p + j
                # W row index: j * n + i  (based on k*n+i where k is col index of C)
                
                U[i * m + k, r] = 1.0
                V[k * p + j, r] = 1.0
                W[j * n + i, r] = 1.0
                r += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
