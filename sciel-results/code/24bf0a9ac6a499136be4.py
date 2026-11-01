"""Rewritten program: exact tensor decomposition for 3x3x6 matrix multiplication."""
import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p  # 54 for 3x3x6
    # U: (n*m) x rank = 9 x 54
    # V: (m*p) x rank = 18 x 54
    # W: (p*n) x rank = 18 x 54
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
