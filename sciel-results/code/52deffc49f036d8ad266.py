"""Matrix multiplication 3x4x4 tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    schoolbook_rank = n * m * p
    
    # Schoolbook decomposition
    best_U = [[0] * schoolbook_rank for _ in range(nm)]
    best_V = [[0] * schoolbook_rank for _ in range(mp)]
    best_W = [[0] * schoolbook_rank for _ in range(pn)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i * m + j][col] = 1
                best_V[j * p + k][col] = 1
                best_W[k * n + i][col] = 1
                col += 1
    
    return {"U": best_U, "V": best_V, "W": best_W}
