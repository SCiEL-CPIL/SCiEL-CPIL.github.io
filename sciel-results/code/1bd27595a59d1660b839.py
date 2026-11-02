for i in range(n):
    for j in range(m):
        for k in range(p):
            U_school[i * m + j, col] = 1
            V_school[j * p + k, col] = 1
            W_school[k * n + i, col] = 1
            col += 1
