     for j in range(m):
         for k in range(p):
             for i in range(n):
                 U[i * m + j][r] = 1
                 W[k * n + i][r] = 1
             V[j * p + k][r] = 1
             r += 1
     