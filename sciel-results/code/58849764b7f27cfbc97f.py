"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 29 decomposition
    # 28 ranks for 7 blocks of 2x2x1
    # 1 rank for 2x1x7
    rank = 29
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Part 1: j in {0, 1}
    # For each k in 0..6, we have a 2x2x1 multiplication
    for k in range(p):
        # 4 terms for 2x2 matrix times 2x1 vector
        # Term 1: A[0,0] B[0,k] -> C[0,k]
        U[0 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # Term 2: A[0,1] B[1,k] -> C[0,k]
        U[0 * m + 1][col] = 1
        V[1 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # Term 3: A[1,0] B[0,k] -> C[1,k]
        U[1 * m + 0][col] = 1
        V[0 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
        # Term 4: A[1,1] B[1,k] -> C[1,k]
        U[1 * m + 1][col] = 1
        V[1 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
        
    # Part 2: j = 2
    # 2x1 matrix times 1x7 matrix -> 2x7 matrix
    # This is rank 1
    # A[i,2] B[2,k] -> C[i,k]
    for i in range(n):
        U[i * m + 2][col] = 1
    for k in range(p):
        V[2 * p + k][col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] = 1
    col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
