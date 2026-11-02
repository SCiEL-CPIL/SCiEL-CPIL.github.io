import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank 15 decomposition
    # R = n * m
    R = n * m
    
    U = np.zeros((n * m, R), dtype=float)
    V = np.zeros((m * p, R), dtype=float)
    W = np.zeros((p * n, R), dtype=float)
    
    for i in range(n):
        for alpha in range(m):
            r = i * m + alpha
            
            # U[i*m + j, r] = delta_{j, alpha}
            for j in range(m):
                if j == alpha:
                    U[i * m + j, r] = 1.0
            
            # V[j'*p + k, r] = delta_{j', alpha}
            for j_prime in range(m):
                for k in range(p):
                    if j_prime == alpha:
                        V[j_prime * p + k, r] = 1.0
            
            # W[k*n + i, r] = 1
            for k in range(p):
                W[k * n + i, r] = 1.0
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
