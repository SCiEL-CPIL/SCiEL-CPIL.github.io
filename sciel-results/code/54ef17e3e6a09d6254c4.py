def construct():
    n, m, p = 3, 3, 6
    # Schoolbook
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try Pan decomposition (rank 23 for 3x3x3, so 46 for 3x3x6)
    # I will try to construct it.
    # If I can't, I return schoolbook.
    
    return {"U": U_sb, "V": V_sb, "W": W_sb}
