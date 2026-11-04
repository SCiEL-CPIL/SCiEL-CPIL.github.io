for i in range(2):
    for k in range(8):
        U[i * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + i][col] = 1
        col += 1
