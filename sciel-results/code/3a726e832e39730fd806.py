for i in range(n):
    for j in range(m):
        for k in range(p):
            U_f[i * m + j][col] = 1
            V_f[j * p + k][col] = 1
            W_f[k * n + i][col] = 1
            col += 1
