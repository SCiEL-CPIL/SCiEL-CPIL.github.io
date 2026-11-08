for j in range(m):
    for k in range(p):
        U[j][col] = 1
        V[j * p + k][col] = 1
        W[k * n][col] = 1
        col += 1
