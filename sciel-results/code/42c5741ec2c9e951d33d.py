   def construct():
       n, m, p = 2, 4, 5
       rank = 36
       U = [[0]*rank for _ in range(n*m)]
       V = [[0]*rank for _ in range(m*p)]
       W = [[0]*rank for _ in range(p*n)]
       
       col = 0
       
       # Define Strassen block parameters
       # (a_r1, a_c1, a_r2, a_c2, b_r1, b_c1, b_r2, b_c2, w_r1, w_c1, w_r2, w_c2)
       # Actually, r1, r2 are always 0, 1 for A and C.
       # c1, c2 are A cols / B rows.
       # k1, k2 are B cols / C cols.
       
       blocks = [
           (0, 1, 0, 1), # Block 1: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
           (2, 3, 0, 1), # Block 2: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
           (0, 1, 2, 3), # Block 3: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
           (2, 3, 2, 3)  # Block 4: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
       ]
       
       for c1, c2, k1, k2 in blocks:
           # r1=0, r2=1
           # U indices: r*m + c
           # V indices: c*p + k
           # W indices: k*n + r
           
           # M1
           U[0*m + c1][col] += 1
           U[1*m + c2][col] += 1
           V[c1*p + k1][col] += 1
           V[c2*p + k2][col] += 1
           W[k1*n + 0][col] += 1
           W[k2*n + 1][col] += 1
           col += 1
           
           # M2
           U[1*m + c1][col] += 1
           U[1*m + c2][col] += 1
           V[c1*p + k1][col] += 1
           W[k1*n + 1][col] += 1
           W[k2*n + 1][col] -= 1
           col += 1
           
           # M3
           U[0*m + c1][col] += 1
           V[c1*p + k2][col] += 1
           V[c2*p + k2][col] -= 1
           W[k2*n + 0][col] += 1
           col += 1
           
           # M4
           U[1*m + c2][col] += 1
           V[c2*p + k1][col] += 1
           V[c1*p + k1][col] -= 1
           W[k1*n + 0][col] += 1
           W[k1*n + 1][col] += 1
           col += 1
           
           # M5
           U[0*m + c1][col] += 1
           U[0*m + c2][col] += 1
           V[c2*p + k2][col] += 1
           W[k1*n + 0][col] -= 1
           W[k2*n + 0][col] += 1
           col += 1
           
           # M6
           U[1*m + c1][col] += 1
           U[0*m + c1][col] -= 1
           V[c1*p + k1][col] += 1
           V[c1*p + k2][col] += 1
           W[k2*n + 1][col] += 1
           col += 1
           
           # M7
           U[0*m + c2][col] += 1
           U[1*m + c2][col] -= 1
           V[c2*p + k1][col] += 1
           V[c2*p + k2][col] += 1
           W[k1*n + 0][col] += 1
           col += 1
           
       # Last column k=4
       for i in range(2):
           for j in range(4):
               U[i*m + j][col] = 1
               V[j*p + 4][col] = 1
               W[4*n + i][col] = 1
               col += 1
               
       return {"U": U, "V": V, "W": W}
   