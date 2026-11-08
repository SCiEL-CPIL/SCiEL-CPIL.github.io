"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 8 decomposition
    rank = 8
    
    # U shape (n*m, rank)
    U = [[0] * rank for _ in range(n * m)]
    # V shape (m*p, rank)
    V = [[0] * rank for _ in range(m * p)]
    # W shape (p*n, rank)
    W = [[0] * rank for _ in range(p * n)]
    
    # Construct U
    # For i=0 (rows 0..3 of U), U[j, j] = 1
    # For i=1 (rows 4..7 of U), U[4+j, 4+j] = 1
    for j in range(m):
        U[j][j] = 1
        U[m + j][m + j] = 1
        
    # Construct V
    # V[j*p + k, c]
    # For c in 0..m-1: V[j*p+k, c] = 1 if j == c
    # For c in m..2m-1: V[j*p+k, c] = 1 if j == c - m
    for j in range(m):
        for k in range(p):
            idx = j * p + k
            V[idx][j] = 1
            V[idx][m + j] = 1
            
    # Construct W
    # W[k*n + i, c]
    # For i=0 (indices 2k): W[2k, c] = 1 for c in 0..m-1
    # For i=1 (indices 2k+1): W[2k+1, c] = 1 for c in m..2m-1
    for k in range(p):
        for c in range(m):
            W[k * n][c] = 1
            W[k * n + 1][m + c] = 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
