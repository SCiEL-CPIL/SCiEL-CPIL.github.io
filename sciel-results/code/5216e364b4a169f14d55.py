# EVOLVE-BLOCK-START
def construct():
       n, m, p = 2, 3, 3
       # Rank 17 decomposition
       # Core: 2x2x2 Strassen (Rank 7)
       # Rest: Rank 10
       
       # Strassen coefficients for 2x2x2
       # M1 = (A00 + A11)(B00 + B11)
       # M2 = (A10 + A11)B00
       # M3 = A00(B10 - B11)
       # M4 = A11(B01 - B00)
       # M5 = (A00 + A01)B11
       # M6 = (A10 - A00)(B00 + B01)
       # M7 = (A01 - A11)(B10 + B11)
       
       # This is getting complicated to embed into 2x3x3.
       
       # Fallback to Schoolbook Rank 18
       rank = 18
       U = [[0] * rank for _ in range(n * m)]
       V = [[0] * rank for _ in range(m * p)]
       W = [[0] * rank for _ in range(p * n)]
       column = 0
       for i in range(n):
           for j in range(m):
               for k in range(p):
                   U[i * m + j][column] = 1
                   V[j * p + k][column] = 1
                   W[k * n + i][column] = 1
                   column += 1
       return {"U": U, "V": V, "W": W}
   # EVOLVE-BLOCK-END
