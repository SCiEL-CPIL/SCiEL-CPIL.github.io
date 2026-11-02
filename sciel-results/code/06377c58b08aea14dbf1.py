"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = p  # 13
    
    # U: shape (n*m, rank) = (4, 13), all ones
    U = [[1] * rank for _ in range(n * m)]
    
    # V: shape (m*p, rank) = (26, 13)
    # V[j*p + k, t] = 1 if k == t, else 0
    V = [[0] * rank for _ in range(m * p)]
    for j in range(m):
        for k in range(p):
            V[j * p + k][k] = 1
    
    # W: shape (p*n, rank) = (26, 13)
    # W[k*n + i, t] = 1 if k == t, else 0
    W = [[0] * rank for _ in range(p * n)]
    for k in range(p):
        for i in range(n):
            W[k * n + i][k] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
