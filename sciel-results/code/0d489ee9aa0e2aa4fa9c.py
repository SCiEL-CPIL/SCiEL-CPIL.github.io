"""
Refined program for matrix multiplication 4x5x6.
Uses a structurally distinct implementation of the schoolbook decomposition.
"""
import numpy as np
import time

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a rank-minimal (currently schoolbook) exact tensor decomposition
    for multiplying a 4 by 5 matrix by a 5 by 6 matrix.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize matrices with zeros
    # Using lists to match the expected output format strictly
    U = [[0.0 for _ in range(rank)] for _ in range(n * m)]
    V = [[0.0 for _ in range(rank)] for _ in range(m * p)]
    W = [[0.0 for _ in range(rank)] for _ in range(p * n)]
    
    # Schoolbook construction
    # Index mapping:
    # U row: i * m + j
    # V row: j * p + k
    # W row: k * n + i
    # Col: c
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx][col] = 1.0
                V[v_idx][col] = 1.0
                W[w_idx][col] = 1.0
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
