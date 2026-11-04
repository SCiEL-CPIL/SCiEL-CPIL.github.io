for i in range(2):
    for j in range(2):
        k = 2
        U[i * 3 + j][col] = 1
        V[j * 5 + k][col] = 1
        W[k * 2 + i][col] = 1
        col += 1
