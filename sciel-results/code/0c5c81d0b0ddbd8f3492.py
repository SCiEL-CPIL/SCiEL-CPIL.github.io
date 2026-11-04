"""Optimal rank decomposition for 2x3 by 3x7 matrix multiplication."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = m * p  # 21
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        for k in range(p):
            col = j * p + k
            # U[i*m + j, col] = 1 for all i
            for i in range(n):
                U[i * m + j][col] = 1
            # V[j*p + k, col] = 1
            V[j * p + k][col] = 1
            # W[k*n + i, col] = 1 for all i
            for i in range(n):
                W[k * n + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
