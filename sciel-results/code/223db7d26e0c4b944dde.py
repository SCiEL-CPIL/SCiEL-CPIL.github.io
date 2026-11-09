"""Tensor decomposition for 3x4 * 4x6 matrix multiplication."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
import time
import random

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank = n * m * p  # 72
    
    # Deadline guard
    deadline = time.time() + 10.0
    
    # Build the multiplication tensor T[i,j,k] = 1 for all i,j,k
    # T is a tensor of shape (n, m, p) where T[i,j,k] = 1
    # We want to find U (n*m x r), V (m*p x r), W (p*n x r) such that
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For now, return the schoolbook solution.
    # This is a valid rank-72 decomposition.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
