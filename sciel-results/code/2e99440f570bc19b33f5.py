"""Rewritten program using explicit loop construction for distinct structure."""
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    R = n * m * p
    
    # Initialize with zeros
    # U: (n*m, R)
    U = [[0] * R for _ in range(n * m)]
    # V: (m*p, R)
    V = [[0] * R for _ in range(m * p)]
    # W: (p*n, R)
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                row_u = i * m + j
                row_v = j * p + k
                row_w = k * n + i
                
                U[row_u][col] = 1
                V[row_v][col] = 1
                W[row_w][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
