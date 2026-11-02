import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Rank R = m * p = 30
    R = m * p
    
    # U shape (n*m, R)
    # U[i, j, r] = 1 if j(r) == j else 0
    # r corresponds to (j_prime, k_prime)
    # r = j_prime * p + k_prime
    U = np.zeros((n * m, R), dtype=int)
    for i in range(n):
        for j in range(m):
            # For a fixed j, U is 1 for all r where j_prime = j
            # These r are j*p + k for k in 0..p-1
            for k_prime in range(p):
                r = j * p + k_prime
                U[i * m + j, r] = 1

    # V shape (m*p, R)
    # V[j, k, r] = 1 if k(r) == k else 0
    V = np.zeros((m * p, R), dtype=int)
    for j in range(m):
        for k in range(p):
            # For a fixed k, V is 1 for all r where k_prime = k
            # These r are j_prime * p + k for j_prime in 0..m-1
            for j_prime in range(m):
                r = j_prime * p + k
                V[j * p + k, r] = 1

    # W shape (p*n, R)
    # W[k, i, r] = 1
    W = np.ones((p * n, R), dtype=int)

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
