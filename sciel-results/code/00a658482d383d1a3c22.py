<<<<<<< SEARCH
    n, m, p = DIMENSIONS
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
=======
    n, m, p = DIMENSIONS
    
    # Total rank: 4 * 11 + 4 * 17 = 44 + 68 = 112
    rank = 112
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Split middle dimension 5 into 2+3
    # A (4x5): A_L is 4x2 (cols 0-1), A_R is 4x3 (cols 2-4)
    # B (5x6): B_T is 2x6 (rows 0-1), B_B is 3x6 (rows 2-4)
    # C = A_L * B_T + A_R * B_B
    
    # Part 1: A_L (4x2) * B_T (2x6) -> 4x6
    # Split A_L rows into 2+2, B_T cols into 3+3
    # 4 blocks of 2x2 * 2x3, each rank 11
    rank_offset = 0
    for i_block in range(2):
        for k_block in range(2):
            A_rows = i_block * 2
            A_cols = 0
            B_rows = 0
            B_cols = k_block * 3
            C_rows = i_block * 2
            C_cols = k_block * 3
            rank_offset = _rank11_2x2x2x3(rank_offset, U, V, W, A_rows, A_cols, B_rows, B_cols, C_rows, C_cols)
    
    # Part 2: A_R (4x3) * B_B (3x6) -> 4x6
    # Split A_R rows into 2+2, B_B cols into 3+3
    # 4 blocks of 2x3 * 3x3, each rank 17
    for i_block in range(2):
        for k_block in range(2):
            A_rows = i_block * 2
            A_cols = 2
            B_rows = 2
            B_cols = k_block * 3
            C_rows = i_block * 2
            C_cols = k_block * 3
            rank_offset = _rank17_2x3x3x3(rank_offset, U, V, W, A_rows, A_cols, B_rows, B_cols, C_rows, C_cols)
    
    return {"U": U, "V": V, "W": W}
>>>>>>> REPLACE
