import numpy as np

def construct():
       n, m, p = 3, 3, 7
       # Schoolbook
       rank_sb = n * m * p
       U_sb = [[0]*rank_sb for _ in range(n*m)]
       V_sb = [[0]*rank_sb for _ in range(m*p)]
       W_sb = [[0]*rank_sb for _ in range(p*n)]
       col = 0
       for i in range(n):
           for j in range(m):
               for k in range(p):
                   U_sb[i*m+j][col] = 1
                   V_sb[j*p+k][col] = 1
                   W_sb[k*n+i][col] = 1
                   col += 1
       
       # Try BCBR Rank 19 for 3x3x3 blocks
       try:
           U_bcbr, V_bcbr, W_bcbr = get_bcbr_3x3()
           # Construct full
           rank_new = 19 + 19 + 9
           U_full = [[0]*rank_new for _ in range(n*m)]
           V_full = [[0]*rank_new for _ in range(m*p)]
           W_full = [[0]*rank_new for _ in range(p*n)]
           
           # Block 1 (cols 0-2)
           for i in range(3):
               for j in range(3):
                   for c in range(19):
                       U_full[i*3+j][c] = U_bcbr[i*3+j][c]
           for j in range(3):
               for k in range(3):
                   for c in range(19):
                       V_full[j*7+k][c] = V_bcbr[j*3+k][c]
           for k in range(3):
               for i in range(3):
                   for c in range(19):
                       W_full[k*3+i][c] = W_bcbr[k*3+i][c]
           
           # Block 2 (cols 3-5)
           offset = 19
           for i in range(3):
               for j in range(3):
                   for c in range(19):
                       U_full[i*3+j][offset+c] = U_bcbr[i*3+j][c]
           for j in range(3):
               for k in range(3):
                   for c in range(19):
                       V_full[j*7+(k+3)][offset+c] = V_bcbr[j*3+k][c]
           for k in range(3):
               for i in range(3):
                   for c in range(19):
                       W_full[(k+3)*3+i][offset+c] = W_bcbr[k*3+i][c]
           
           # Block 3 (col 6) - Schoolbook for 3x3x1
           offset = 38
           idx = offset
           for i in range(3):
               for j in range(3):
                   U_full[i*3+j][idx] = 1
                   V_full[j*7+6][idx] = 1
                   W_full[6*3+i][idx] = 1
                   idx += 1
           
           # Verify
           if verify_decomposition(U_full, V_full, W_full, n, m, p):
               return {"U": U_full, "V": V_full, "W": W_full}
           else:
               return {"U": U_sb, "V": V_sb, "W": W_sb}
       except Exception:
           return {"U": U_sb, "V": V_sb, "W": W_sb}
