for i in range(n):
    for k in range(p):
        U[i * m + 2][r] = 1
        V[2 * p + k][r] = 1
        W[k * n + i][r] = 1
        r += 1
