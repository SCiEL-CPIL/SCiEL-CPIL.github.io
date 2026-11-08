   def construct():
       n, m, p = 2, 3, 3
       rank = 15
       U = np.zeros((n*m, rank), dtype=int)
       V = np.zeros((m*p, rank), dtype=int)
       W = np.zeros((p*n, rank), dtype=int)
       
       # Indices:
       # U[i*m+j, c]
       # V[j*p+k, c]
       # W[k*n+i, c]
       
       # Term 0
       U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
       # Term 1
       U[1, 1] = 1; V[3, 1] = 1; W[1, 1] = 1
       # Term 2
       U[2, 2] = 1; V[6, 2] = 1; W[2, 2] = 1
       # Term 3
       U[3, 3] = 1; V[0, 3] = 1; W[3, 3] = 1
       # Term 4
       U[4, 4] = 1; V[3, 4] = 1; W[4, 4] = 1
       # Term 5
       U[5, 5] = 1; V[6, 5] = 1; W[5, 5] = 1
       # Term 6
       U[0, 6] = 1; V[1, 6] = 1; W[6, 6] = 1
       # Term 7
       U[1, 7] = 1; V[4, 7] = 1; W[7, 7] = 1
       # Term 8
       U[2, 8] = 1; V[7, 8] = 1; W[8, 8] = 1
       # Term 9
       U[3, 9] = 1; V[1, 9] = 1; W[9, 9] = 1
       # Term 10
       U[4, 10] = 1; V[4, 10] = 1; W[10, 10] = 1
       # Term 11
       U[5, 11] = 1; V[7, 11] = 1; W[11, 11] = 1
       # Term 12
       U[0, 12] = 1; V[2, 12] = 1; W[12, 12] = 1
       # Term 13
       U[1, 13] = 1; V[5, 13] = 1; W[13, 13] = 1
       # Term 14
       U[2, 14] = 1; V[8, 14] = 1; W[14, 14] = 1
       
       return {"U": U, "V": V, "W": W}
   