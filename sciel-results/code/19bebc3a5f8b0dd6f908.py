import numpy as np
import time

def construct():
    n, m, p = 2, 3, 5
    
    # Dimensions
    dim_U = n * m  # 6
    dim_V = m * p  # 15
    dim_W = p * n  # 10
    
    # Schoolbook rank
    rank = n * m * p  # 30
    
    # Initialize matrices
    U = [[0] * rank for _ in range(dim_U)]
    V = [[0] * rank for _ in range(dim_V)]
    W = [[0] * rank for _ in range(dim_W)]
    
    # Fill in the schoolbook decomposition
    # T[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
    # For schoolbook, each term r corresponds to a specific (i, j, k)
    # u = i * m + j
    # v = j * p + k
    # w = k * n + i
    # U[u, r] = 1
    # V[v, r] = 1
    # W[w, r] = 1
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                
                U[u][r] = 1
                V[v][r] = 1
                W[w][r] = 1
                r += 1
                
    return {"U": U, "V": V, "W": W}
