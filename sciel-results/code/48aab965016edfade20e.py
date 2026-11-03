"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=4
    # Rank 16 decomposition
    # Split output columns into two blocks of 2.
    # Block 1: Cols 0,1 of C. Uses Cols 0,1 of B.
    # Block 2: Cols 2,3 of C. Uses Cols 2,3 of B.
    
    rank = 16
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Block 1: k in {0, 1}
    for j in range(m):
        for k in range(2):
            r = j * 2 + k
            for i in range(n):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                
    # Block 2: k in {2, 3}
    for j in range(m):
        for k in range(2, p):
            r = 8 + j * 2 + (k - 2)
            for i in range(n):
                U[i * m + j][r] = 1
                V[j * p + k][r] = 1
                W[k * n + i][r] = 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
