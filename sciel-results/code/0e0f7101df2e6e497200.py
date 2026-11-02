import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    DEADLINE = 140  # seconds, safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    total_terms = n * m * p  # 96
    
    # Schoolbook decomposition
    def schoolbook():
        rank = total_terms
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m + j, col] = 1
                    V[j*p + k, col] = 1
                    W[k*n + i, col] = 1
                    col += 1
        return U, V, W
    
    # Try to find a better decomposition
    # The tensor T[i,j,k] = 1 for all i,j,k
    # We need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Let's try to use a randomized approach or structured decomposition
    # For now, return schoolbook as baseline
    U, V, W = schoolbook()
    
    # Convert to lists
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
