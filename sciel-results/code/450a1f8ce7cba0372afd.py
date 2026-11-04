import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Use trivial decomposition for all blocks to ensure correctness
    # Rank will be 2*4*5 = 40
    rank = 40
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U[i*m + j, col] = 1
                U[i * m + j][col] = 1
                # V[j*p + k, col] = 1
                V[j * p + k][col] = 1
                # W[k*n + i, col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
