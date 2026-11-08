     for i in range(n):
         for j in range(m):
             for k in range(p):
                 U[i * m + j][column] = 1
                 V[j * p + k][column] = 1
                 W[k * n + i][column] = 1
                 column += 1
     