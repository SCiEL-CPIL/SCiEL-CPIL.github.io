    col = 0
    for i in range(4):
        for j in range(4):
            U_441[i * 4 + j, col] = 1
            V_441[j, col] = 1
            W_441[i, col] = 1
            col += 1
