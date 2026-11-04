for i in range(n):  # 2
    for k in range(3):  # 3 column blocks
        for t in range(2):  # 2 entries in 1x2 block
            for j in range(m):  # 5
                U12[i * m + j][col] = 1
                V12[j * p + (k * 2 + t)][col] = 1
            W12[(k * 2 + t) * n + i][col] = 1
            col += 1
