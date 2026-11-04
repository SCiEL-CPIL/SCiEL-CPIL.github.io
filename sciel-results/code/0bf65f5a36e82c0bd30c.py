# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 3, 4
    rank = 32
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    # 23 terms for 3x3x3
    # I will use the schoolbook for 3x3x3 (27 terms) to be safe?
    # No, that's 27+9=36.
    
    # I will use the Strassen 2x2x2 (7 terms) for the top-left 2x2x2 block?
    # And schoolbook for the rest?
    # This is a valid decomposition?
    # If I partition the indices, I can use different decompositions for different blocks.
    # But the blocks overlap in the middle indices.
    
    # I will stick to Rank 36.
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1.0
                V[j * p + k][column] = 1.0
                W[k * n + i][column] = 1.0
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
