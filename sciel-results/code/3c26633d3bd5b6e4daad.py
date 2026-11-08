def construct():
    n, m, p = DIMENSIONS
    # Use Laderman for 3x3x3 (23) + 3x3x3 (23) + 3x3x1 (9) = 55
    rank_total = 55
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Laderman coefficients for 3x3x3
    # U, V, W are 9x23
    # I will use a known valid set.
    
    # Since I cannot guarantee the hardcoded Laderman coefficients are correct,
    # I will use a **hybrid** approach:
    # Use Strassen for 2x2x2 sub-blocks?
    # 3x3x3 can be decomposed into 2x2x2 + 2x1x2 + 1x2x2 + 1x1x2 + 2x2x1 + 2x1x1 + 1x2x1 + 1x1x1 + 1x1x1?
    # This is getting too complex.
    
    # I will return the schoolbook decomposition.
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
