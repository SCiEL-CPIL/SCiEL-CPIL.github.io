import time

DIMENSIONS = [2, 5, 5]

def construct():
    """
    Returns the schoolbook decomposition for 2x5 x 5x5 matrix multiplication.
    Rank = 2 * 5 * 5 = 50.
    """
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Initialize U, V, W with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Fill in the schoolbook decomposition
    # U[i*m + j, c] = 1
    # V[j*p + k, c] = 1
    # W[k*n + i, c] = 1
    # for each (i, j, k) assigned to column c
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_u = i * m + j
                idx_v = j * p + k
                idx_w = k * n + i
                
                U[idx_u][col] = 1
                V[idx_v][col] = 1
                W[idx_w][col] = 1
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
