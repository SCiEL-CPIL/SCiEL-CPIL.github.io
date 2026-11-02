    for i in range(n):
        for j_local in range(2):
            j = j_local + 1
            U[i * 3 + j][col] = 1
            V[j * 5 + 2][col] = 1
            W[2 * 2 + i][col] = 1
            col += 1
