"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """
    Construct the rank-minimal exact tensor decomposition for 2x2 * 2x15 matrix multiplication.
    
    Uses schoolbook decomposition: rank = 4 * 15 = 60
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition: rank = 4 * p = 60
    total_rank = 4 * p
    
    U = [[0] * total_rank for _ in range(n * m)]  # 4 x 60
    V = [[0] * total_rank for _ in range(m * p)]  # 30 x 60
    W = [[0] * total_rank for _ in range(p * n)]  # 30 x 60
    
    # For each column l of B, we have 4 terms:
    # Term 4l:   A[0,0] * B[0,l] -> C[0,l]
    # Term 4l+1: A[0,1] * B[1,l] -> C[0,l]
    # Term 4l+2: A[1,0] * B[0,l] -> C[1,l]
    # Term 4l+3: A[1,1] * B[1,l] -> C[1,l]
    
    for l in range(p):
        # Term 4l: A[0,0] * B[0,l] -> C[0,l]
        r = 4 * l
        U[0 * m + 0][r] = 1  # A[0,0]
        V[0 * p + l][r] = 1  # B[0,l]
        W[l * n + 0][r] = 1  # C[0,l]
        
        # Term 4l+1: A[0,1] * B[1,l] -> C[0,l]
        r = 4 * l + 1
        U[0 * m + 1][r] = 1  # A[0,1]
        V[1 * p + l][r] = 1  # B[1,l]
        W[l * n + 0][r] = 1  # C[0,l]
        
        # Term 4l+2: A[1,0] * B[0,l] -> C[1,l]
        r = 4 * l + 2
        U[1 * m + 0][r] = 1  # A[1,0]
        V[0 * p + l][r] = 1  # B[0,l]
        W[l * n + 1][r] = 1  # C[1,l]
        
        # Term 4l+3: A[1,1] * B[1,l] -> C[1,l]
        r = 4 * l + 3
        U[1 * m + 1][r] = 1  # A[1,1]
        V[1 * p + l][r] = 1  # B[1,l]
        W[l * n + 1][r] = 1  # C[1,l]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
