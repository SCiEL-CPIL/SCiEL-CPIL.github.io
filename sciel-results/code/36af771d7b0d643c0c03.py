import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Standard matrix multiplication tensor decomposition
    # C[i, k] = sum_j A[i, j] B[j, k]
    # The tensor is a sum of rank-1 terms:
    # T = sum_{i,j,k} E_{i,j} (of A) tensor E_{j,k} (of B) tensor E_{i,k} (of C)
    # Note: The output index for W is given as k*n + i.
    
    R = n * m * p
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Index in U (A matrix): row i, col j -> i * m + j
                a = i * m + j
                # Index in V (B matrix): row j, col k -> j * p + k
                b = j * p + k
                # Index in W (C matrix): row k, col i -> k * n + i
                c = k * n + i
                
                U[a, r] = 1
                V[b, r] = 1
                W[c, r] = 1
                r += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
