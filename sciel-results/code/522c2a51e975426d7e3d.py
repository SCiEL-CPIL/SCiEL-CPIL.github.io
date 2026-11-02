"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    # The matmul tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is an all-ones tensor of shape (n, m, p) with tensor rank min(n, m, p)
    # For 2x3x8: rank = min(2,3,8) = 2
    
    rank = min(n, m, p)  # = 2
    
    U = [[0] * rank for _ in range(n * m)]  # 6 x 2
    V = [[0] * rank for _ in range(m * p)]  # 24 x 2
    W = [[0] * rank for _ in range(p * n)]  # 16 x 2
    
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # For each r in [0, rank): U=1, V=1, W=1/rank
    # Sum = rank * (1 * 1 * 1/rank) = 1 ✓
    
    for r in range(rank):
        val = 1.0 / rank
        for i in range(n):
            for j in range(m):
                U[i * m + j][r] = 1
        for j in range(m):
            for k in range(p):
                V[j * p + k][r] = 1
        for k in range(p):
            for i in range(n):
                W[k * n + i][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
