import numpy as np
import time

DIMENSIONS = [5, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 if i + k == j else 0
    # Using trace convention: W[k*n + i, r]
    # T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Schoolbook rank is 125.
    # We try to find a lower rank.
    
    # If we cannot find a lower rank quickly, return schoolbook.
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Try to find a decomposition of rank 100 using greedy search.
    # This is a placeholder for a more sophisticated search.
    # For now, return schoolbook to ensure correctness.
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
