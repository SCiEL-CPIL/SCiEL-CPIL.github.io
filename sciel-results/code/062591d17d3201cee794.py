"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Use schoolbook algorithm for 3x3x6
    # Rank is n*m*p = 3*3*6 = 54
    rank = n * m * p
    
    U = [[0] * rank for _ in range(n * m)]  # (9, 54)
    V = [[0] * rank for _ in range(m * p)]  # (18, 54)
    W = [[0] * rank for _ in range(p * n)]  # (18, 54)
    
    col = 0
    # Standard bilinear form: C[i,k] = sum_j A[i,j] B[j,k]
    # Term for (i,j,k): U[i*m+j] = 1, V[j*p+k] = 1, W[k*n+i] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
