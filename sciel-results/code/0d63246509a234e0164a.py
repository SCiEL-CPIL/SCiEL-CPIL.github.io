    for i in range(n):
        for j in range(m):
            U[i*m+j][col] = 1
            V[j*p+8][col] = 1
            W[8*n+i][col] = 1
            col += 1
