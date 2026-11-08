   for k0, k1 in k_pairs:
       for k in [k0, k1]:
           # C_0,k = A_0,4 B_4,k
           U[0*m + 4][col] += 1
           V[4*p + k][col] += 1
           W[k*n + 0][col] += 1
           col += 1
           
           # C_1,k = A_1,4 B_4,k
           U[1*m + 4][col] += 1
           V[4*p + k][col] += 1
           W[k*n + 1][col] += 1
           col += 1
   