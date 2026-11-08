   def construct():
       n, m, p = 2, 5, 6
       rank = 0
       U = []
       V = []
       W = []
       
       # Helper to append a rank-1 term
       def add_term(u, v, w):
           U.append(u)
           V.append(v)
           W.append(w)
           
       # 1. Strassen for 2x2x2
       # Returns list of (u, v, w) where u is len 4, v len 4, w len 4
       # Indices: u[i*2+j], v[j*2+k], w[k*2+i]
       def get_strassen_terms():
           # M1 = (A11+A22)(B11+B22)
           # M2 = (A21+A22)B11
           # M3 = A11(B12-B22)
           # M4 = A22(B21-B11)
           # M5 = (A11+A12)B22
           # M6 = (A21-A11)(B11+B12)
           # M7 = (A12-A22)(B21+B22)
           
           terms = []
           
           # M1
           u1 = [1, 0, 0, 1] # A11+A22
           v1 = [1, 1, 1, 1] # B11+B22 (Wait, B is 2x2. B11, B12, B21, B22. Indices j*2+k)
           # B indices: 0: B11, 1: B12, 2: B21, 3: B22.
           # v1 = [1, 1, 1, 1]
           w1 = [1, 0, 0, 1] # C11, C22 get +1?
           # C11 = M1 + ...
           # C22 = M1 - ...
           # So w1 = [1, 0, 0, 1] (C11, C12, C21, C22) -> C11=1, C22=1.
           terms.append((u1, v1, w1))
           
           # M2
           u2 = [0, 0, 1, 1] # A21+A22
           v2 = [1, 0, 0, 0] # B11
           w2 = [0, 0, 1, -1] # C21=1, C22=-1
           terms.append((u2, v2, w2))
           
           # M3
           u3 = [1, 0, 0, 0] # A11
           v3 = [0, 1, 0, -1] # B12-B22
           w3 = [0, 1, 0, 0] # C12=1
           terms.append((u3, v3, w3))
           
           # M4
           u4 = [0, 0, 0, 1] # A22
           v4 = [0, 0, 1, -1] # B21-B11
           w4 = [1, 0, 1, 0] # C11=1, C21=1
           terms.append((u4, v4, w4))
           
           # M5
           u5 = [1, 1, 0, 0] # A11+A12
           v5 = [0, 0, 0, 1] # B22
           w5 = [-1, 0, 0, 1] # C11=-1, C22=1
           terms.append((u5, v5, w5))
           
           # M6
           u6 = [0, 0, 1, -1] # A21-A11
           v6 = [1, 1, 0, 0] # B11+B12
           w6 = [0, 0, 0, 1] # C22=1
           terms.append((u6, v6, w6))
           
           # M7
           u7 = [0, 1, 0, -1] # A12-A22
           v7 = [0, 0, 1, 1] # B21+B22
           w7 = [1, 0, 0, 0] # C11=1
           terms.append((u7, v7, w7))
           
           return terms
       
       # Global dimensions
       # U size: n*m = 10
       # V size: m*p = 30
       # W size: p*n = 12
       
       U = [[0]*10 for _ in range(57)] # Pre-allocate? Or just build list of lists
       V = [[0]*30 for _ in range(57)]
       W = [[0]*12 for _ in range(57)]
       
       col = 0
       
       # Part 1: j in [0, 2), k in [0, 6)
       # 3 blocks of 2x2x2
       strassen_terms = get_strassen_terms()
       for k_block in range(3):
           k_start = 2 * k_block
           for u, v, w in strassen_terms:
               # Map local indices to global
               # Local i: 0,1 -> Global i: 0,1
               # Local j: 0,1 -> Global j: 0,1
               # Local k: 0,1 -> Global k: k_start, k_start+1
               
               U[col] = [0]*10
               V[col] = [0]*30
               W[col] = [0]*12
               
               for li in range(2):
                   for lj in range(2):
                       U[col][li*2 + lj] = u[li*2 + lj]
                       
               for lj in range(2):
                   for lk in range(2):
                       V[col][lj*6 + (k_start + lk)] = v[lj*2 + lk]
                       
               for lk in range(2):
                   for li in range(2):
                       W[col][(k_start + lk)*2 + li] = w[lk*2 + li]
                       
               col += 1
               
       # Part 2: j in [2, 5), k in [0, 6)
       # 2 blocks of 2x3x3
       # Schoolbook for 2x3x3
       for k_block in range(2):
           k_start = 3 * k_block
           for li in range(2):
               for lj in range(3):
                   for lk in range(3):
                       U[col] = [0]*10
                       V[col] = [0]*30
                       W[col] = [0]*12
                       
                       U[col][li*5 + (2 + lj)] = 1
                       V[col][(2 + lj)*6 + (k_start + lk)] = 1
                       W[col][(k_start + lk)*2 + li] = 1
                       
                       col += 1
                       
       return {"U": U, "V": V, "W": W}
   