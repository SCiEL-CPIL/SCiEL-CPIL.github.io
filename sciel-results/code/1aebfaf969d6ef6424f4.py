   def construct():
       n, m, p = 3, 4, 5
       dim_u = n * m
       dim_v = m * p
       dim_w = p * n
       
       U_cols = []
       V_cols = []
       W_cols = []
       
       # Add Strassen blocks
       # ... (same as before) ...
       
       # Add Term A (k=4, i in {0,1})
       U_A = np.zeros(dim_u)
       V_A = np.zeros(dim_v)
       W_A = np.zeros(dim_w)
       
       for i in range(2):
           for j in range(4):
               U_A[i*m + j] = 1
       
       for j in range(4):
           V_A[j*p + 4] = 1
           
       for i in range(2):
           W_A[4*n + i] = 1
           
       U_cols.append(U_A)
       V_cols.append(V_A)
       W_cols.append(W_A)
       
       # Add Term B (i=2)
       U_B = np.zeros(dim_u)
       V_B = np.zeros(dim_v)
       W_B = np.zeros(dim_w)
       
       for j in range(4):
           U_B[2*m + j] = 1
           
       for j in range(4):
           for k in range(5):
               V_B[j*p + k] = 1
               
       for k in range(5):
           W_B[k*n + 2] = 1
           
       U_cols.append(U_B)
       V_cols.append(V_B)
       W_cols.append(W_B)
       
       U = np.column_stack(U_cols)
       V = np.column_stack(V_cols)
       W = np.column_stack(W_cols)
       
       # Verification
       # ...
       
       return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
   