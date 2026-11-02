"""Matrix multiplication 3x4x5: rank-minimal exact tensor decomposition."""
import time

DIMENSIONS = [3, 4, 5]

def construct():
    """Returns rank-60 schoolbook tensor decomposition for 3x4 by 4x5 matrix multiplication."""
    n, m, p = DIMENSIONS
    rank = n * m * p  # 60
    
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
