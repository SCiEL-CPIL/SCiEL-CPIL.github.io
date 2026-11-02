"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=3, p=9
    # Decompose into 3 blocks of 2x3 x 3x3
    num_blocks = p // m  # 9 // 3 = 3
    rank_per_block = n * m * m  # 2*3*3 = 6
    rank = num_blocks * rank_per_block  # 3*6 = 18
    
    U = [[0] * rank for _ in range(n * m)]  # shape (6, 18)
    V = [[0] * rank for _ in range(m * p)]  # shape (27, 18)
    W = [[0] * rank for _ in range(p * n)]  # shape (18, 18)
    
    col = 0
    for t in range(num_blocks):
        for i in range(n):
            for j in range(m):
                for k in range(m):
                    # A[i, j] contributes
                    U[i * m + j][col] = 1
                    # B[j, t*m + k] contributes
                    V[j * p + (t * m + k)][col] = 1
                    # Output C[i, t*m + k], index in W is (t*m+k)*n + i
                    W[(t * m + k) * n + i][col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
