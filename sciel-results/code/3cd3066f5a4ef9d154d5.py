col = 0
for i in range(1):
    for j in range(2):
        for k in range(2):
            U[i * 2 + j][col] = 1
            V[j * 2 + k][col] = 1
            W[k * 1 + 0][col] = 1
            col += 1
