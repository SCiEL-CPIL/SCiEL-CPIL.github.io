   def construct():
       n, m, p = 2, 2, 6
       rank = 21
       U = [[0] * rank for _ in range(n * m)]
       V = [[0] * rank for _ in range(m * p)]
       W = [[0] * rank for _ in range(p * n)]
       
       # Strassen coefficients for 2x2
       # u[i*m+j], v[j*p+k], w[k*n+i]
       # 7 terms per block
       
       terms = []
       
       # Term 1
       u1 = [1, 0, 0, 1] # a00, a11
       v1 = [1, 0, 0, 1] # b00, b11
       w1 = [1, 0, 0, 1] # c00, c11
       terms.append((u1, v1, w1))
       
       # Term 2
       u2 = [0, 0, 1, 1] # a10, a11
       v2 = [1, 0, 0, 0] # b00
       w2 = [0, 0, 1, -1] # c10, c11
       terms.append((u2, v2, w2))
       
       # Term 3
       u3 = [1, 0, 0, 0] # a00
       v3 = [0, 0, 1, -1] # b10, b11
       w3 = [0, 1, 0, 1] # c01, c11
       terms.append((u3, v3, w3))
       
       # Term 4
       u4 = [0, 0, 0, 1] # a11
       v4 = [0, 1, -1, 0] # b01, b00
       w4 = [1, 0, 1, 0] # c00, c10
       terms.append((u4, v4, w4))
       
       # Term 5
       u5 = [1, 0, 1, 0] # a00, a10
       v5 = [0, 0, 0, 1] # b11
       w5 = [-1, 1, 0, 0] # c00, c01
       terms.append((u5, v5, w5))
       
       # Term 6
       u6 = [0, 0, 0, 1] # a11
       v6 = [1, 0, 1, 0] # b00, b10
       w6 = [0, 0, 0, 1] # c11
       terms.append((u6, v6, w6))
       
       # Term 7
       u7 = [0, 1, 0, -1] # a01, a11
       v7 = [0, 1, 1, 0] # b01, b11
       w7 = [1, -1, 0, 0] # c00, c01
       terms.append((u7, v7, w7))
       
       for l in range(3):
           for r in range(7):
               col = l * 7 + r
               u, v, w = terms[r]
               # u is indexed by (i, j) -> i*2+j
               # v is indexed by (j, k_local) -> j*2+k_local
               # w is indexed by (k_local, i) -> k_local*2+i
               
               for idx in range(4):
                   i = idx // 2
                   j = idx % 2
                   U[i*m+j][col] = u[idx]
                   
                   j_v = idx // 2
                   k_v = idx % 2
                   V[j_v*m + (2*l + k_v)][col] = v[idx]
                   
                   k_w = idx // 2
                   i_w = idx % 2
                   W[(2*l + k_w)*n + i_w][col] = w[idx]
       
       return {"U": U, "V": V, "W": W}
   