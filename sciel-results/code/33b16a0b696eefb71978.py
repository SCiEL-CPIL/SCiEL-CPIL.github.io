"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import sys

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Dimensions
    NM = n * m  # 12
    MP = m * p  # 24
    PN = p * n  # 18
    
    # Rank 24 construction
    R = MP  # 24
    
    U = np.zeros((NM, R))
    V = np.zeros((MP, R))
    W = np.zeros((PN, R))
    
    # V is identity matrix
    for c in range(R):
        V[c, c] = 1.0
        
    # Fill U and W
    for c in range(R):
        # c corresponds to j*p + k
        j = c // p
        k = c % p
        
        # U indices: i*m + j for i in 0..n-1
        for i in range(n):
            u_idx = i * m + j
            U[u_idx, c] = 1.0
            
        # W indices: k*n + i for i in 0..n-1
        for i in range(n):
            w_idx = k * n + i
            W[w_idx, c] = 1.0
            
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
# EVOLVE-BLOCK-END
