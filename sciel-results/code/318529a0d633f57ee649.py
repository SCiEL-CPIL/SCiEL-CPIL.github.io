import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    # Fill with schoolbook decomposition
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Optional: Verify a few entries to ensure correctness
    # The tensor T[i,j,k] should be 1 for all i,j,k
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Since supports are disjoint, each (i,j,k) has exactly one c with 1s, so sum is 1.
    # This is structurally guaranteed.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
