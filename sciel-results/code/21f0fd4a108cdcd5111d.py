"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using 5 Strassen blocks (rank 7 each) and 1 Trivial block (rank 4)
    # Total rank = 5 * 7 + 4 = 39
    rank = 39
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    # Process 5 pairs of columns (0-9) using Strassen decomposition
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Term 1: (A00 + A11)(B0k0 + B1k1)
        # Contributes to C[k0, 0] and C[k1, 1]
        U[0, col] = 1
        U[3, col] = 1
        V[0 * p + k0, col] = 1
        V[1 * p + k1, col] = 1
        W[k0 * n + 0, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1
        
        # Term 2: (A10 + A11)B0k0
        # Contributes to C[k0, 1]
        U[2, col] = 1
        U[3, col] = 1
        V[0 * p + k0, col] = 1
        W[k0 * n + 1, col] = 1
        col += 1
        
        # Term 3: A00(B0k1 - B1k1)
        # Contributes to C[k1, 0]
        U[0, col] = 1
        V[0 * p + k1, col] = 1
        V[1 * p + k1, col] = -1
        W[k1 * n + 0, col] = 1
        col += 1
        
        # Term 4: A11(B1k0 - B0k0)
        # Contributes to C[k0, 0] and C[k0, 1]
        U[3, col] = 1
        V[1 * p + k0, col] = 1
        V[0 * p + k0, col] = -1
        W[k0 * n + 0, col] = 1
        W[k0 * n + 1, col] = 1
        col += 1
        
        # Term 5: (A00 + A01)B1k1
        # Contributes to -C[k0, 0] and C[k1, 0]
        U[0, col] = 1
        U[1, col] = 1
        V[1 * p + k1, col] = 1
        W[k0 * n + 0, col] = -1
        W[k1 * n + 0, col] = 1
        col += 1
        
        # Term 6: (A10 - A00)(B0k0 + B0k1)
        # Contributes to C[k1, 1]
        U[2, col] = 1
        U[0, col] = -1
        V[0 * p + k0, col] = 1
        V[0 * p + k1, col] = 1
        W[k1 * n + 1, col] = 1
        col += 1
        
        # Term 7: (A01 - A11)(B1k0 + B1k1)
        # Contributes to -C[k0, 0]
        U[1, col] = 1
        U[3, col] = -1
        V[1 * p + k0, col] = 1
        V[1 * p + k1, col] = 1
        W[k0 * n + 0, col] = -1
        col += 1

    # Process last column (10) using Trivial decomposition
    k = 10
    
    # Term 1: A00 * B0k
    U[0, col] = 1
    V[0 * p + k, col] = 1
    W[k * n + 0, col] = 1
    col += 1
    
    # Term 2: A01 * B1k
    U[1, col] = 1
    V[1 * p + k, col] = 1
    W[k * n + 0, col] = 1
    col += 1
    
    # Term 3: A10 * B0k
    U[2, col] = 1
    V[0 * p + k, col] = 1
    W[k * n + 1, col] = 1
    col += 1
    
    # Term 4: A11 * B1k
    U[3, col] = 1
    V[1 * p + k, col] = 1
    W[k * n + 1, col] = 1
    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
