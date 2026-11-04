    # Part 3: 2x4x1 (k=6)
    # 4 rank-1 terms (one for each j)
    for j in range(4):
        U[0*m + j][col] = 1.0
        U[1*m + j][col] = 1.0
        V[j*p + 6][col] = 1.0
        W[6*n + 0][col] = 1.0
        W[6*n + 1][col] = 1.0
        col += 1
    