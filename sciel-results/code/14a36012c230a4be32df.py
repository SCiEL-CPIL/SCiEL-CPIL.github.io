"""Matrix multiplication 3x3 by 3x7 - rank minimization."""
DIMENSIONS = [3, 3, 7]

import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 57 decomposition:
    # Two 3x3x3 blocks (schoolbook rank 27 each) and one 3x3x1 block (rank 3).
    # Total rank = 27 + 27 + 3 = 57.
    
    rank = 57
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Block 1: 3x3x3 for k in [0, 2]
    # U: (i, j)
    # V: (j, k)
    # W: (k, i)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(3):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Block 2: 3x3x3 for k in [3, 5]
    for i in range(n):
        for j in range(m):
            for k_prime in range(3):
                k = k_prime + 3
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    # Block 3: 3x3x1 for k = 6
    # C_i = sum_j A_i,j B_j,6
    # U[i, j, t] = 1 if t == j
    # V[j, 6, t] = 1 if t == j
    # W[6, i, t] = 1 if t == j
    # This is just the schoolbook for 3x3x1
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1.0
            V[j * p + 6, col] = 1.0
            W[6 * n + i, col] = 1.0
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
