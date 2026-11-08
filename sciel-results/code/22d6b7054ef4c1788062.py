for i in range(n):
    for a in range(m):
        for k in range(p):
            U[i * m + a][col] = 1
            V[a * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
