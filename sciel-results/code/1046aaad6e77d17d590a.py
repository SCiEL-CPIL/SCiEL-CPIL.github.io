for i in range(2):
    for j in range(2):
        col = 7 + i*2 + j
        U_full[i*2 + j][col] = 1
        V_full[j*3 + 2][col] = 1
        W_full[2*2 + i][col] = 1
