"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank-m decomposition: split the m=4 inner dimension into m=4 blocks of size 1.
    # For each j in 0..m-1, one rank-1 term that picks out A[i,j] and B[j,k].
    rank = m  # = 4
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for j in range(m):
        c = j  # column index
        for i in range(n):
            U[i * m + j][c] = 1
        for k in range(p):
            V[j * p + k][c] = 1
        for i in range(n):
            for k in range(p):
                W[k * n + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
