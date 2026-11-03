"""Matrix multiplication 3x3x3 - schoolbook rank-27 decomposition."""
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns rank-27 schoolbook decomposition for 3x3 matrix multiplication.
    
    U: (9, 27), V: (9, 27), W: (9, 27)
    W uses convention W[k*n+i, r]
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 3, 3, 3
    rank = n * m * p  # 27
    
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
