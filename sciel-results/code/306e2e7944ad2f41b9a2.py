"""Matrix multiplication 2x3x10 rank-minimal decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def construct():
    """
    Rank-55 decomposition using Strassen-like reduction.
    
    Split output into 5 pairs of columns. For each pair, compute 2x3 x 3x2 block.
    Use rank-7 decomposition for 2x3 x 3x2 (Strassen-based), total rank = 5*7 = 35? 
    No, that's not right either.
    
    Let me just do schoolbook rank 60 correctly first, then optimize.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=10
    rank = n * m * p  # 60
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
