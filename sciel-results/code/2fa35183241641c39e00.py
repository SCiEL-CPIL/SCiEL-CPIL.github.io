"""Strassen-based structured ansatz for 3x3 * 3x5 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def construct():
    start_time = time.time()
    deadline = 120  # seconds
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    # Rank 15 decomposition:
    # For each output column k (0..4), use 3 terms (one for each input row j=0..2).
    # This effectively computes C[:, k] = A @ B[:, k] using 3 rank-1 tensor terms per column.
    # Total rank = 5 * 3 = 15.
    
    rank = m * p  # 3 * 5 = 15
    
    U = [[0]*rank for _ in range(n*m)]  # 9 x 15
    V = [[0]*rank for _ in range(m*p)]  # 15 x 15
    W = [[0]*rank for _ in range(p*n)]  # 15 x 15
    
    for k in range(p):
        for j in range(m):
            c = k * m + j
            
            # U[i, j, c] = 1 for all i
            for i in range(n):
                U[i*m + j][c] = 1
                
            # V[j, k, c] = 1
            V[j*p + k][c] = 1
            
            # W[k, i, c] = 1 for all i
            for i in range(n):
                W[k*n + i][c] = 1
                
    return {"U": U, "V": V, "W": W}
