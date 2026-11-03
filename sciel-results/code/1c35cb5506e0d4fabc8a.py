   # A21*B12: 1x2*2x1, rank 4
   for j in range(2):
       U[2 * 3 + j, col] = 1.0
       V[j * 3 + 2, col] = 1.0
       W[2 * 3 + 2, col] = 1.0
       col += 1
   