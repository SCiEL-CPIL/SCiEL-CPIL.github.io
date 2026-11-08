   def kron_decomp(d1, d2):
       # d1: U1, V1, W1 for n1, m1, p1, rank r1
       # d2: U2, V2, W2 for n2, m2, p2, rank r2
       n1, m1, p1 = d1['dims']
       n2, m2, p2 = d2['dims']
       r1 = d1['rank']
       r2 = d2['rank']
       
       U1, V1, W1 = d1['U'], d1['V'], d1['W']
       U2, V2, W2 = d2['U'], d2['V'], d2['W']
       
       n = n1 * n2
       m = m1 * m2
       p = p1 * p2
       r = r1 * r2
       
       U = [[0]*r for _ in range(n*m)]
       V = [[0]*r for _ in range(m*p)]
       W = [[0]*r for _ in range(p*n)]
       
       for r1_idx in range(r1):
           for r2_idx in range(r2):
               r_idx = r1_idx * r2 + r2_idx
               for i1 in range(n1):
                   for j1 in range(m1):
                       u1_val = U1[i1*m1+j1][r1_idx]
                       if u1_val == 0: continue
                       for i2 in range(n2):
                           for j2 in range(m2):
                               u2_val = U2[i2*m2+j2][r2_idx]
                               if u2_val == 0: continue
                               i = i1*n2 + i2
                               j = j1*m2 + j2
                               U[i*m+j][r_idx] += u1_val * u2_val
       
       # Similar for V and W
       # V indices: j, k
       for r1_idx in range(r1):
           for r2_idx in range(r2):
               r_idx = r1_idx * r2 + r2_idx
               for j1 in range(m1):
                   for k1 in range(p1):
                       v1_val = V1[j1*p1+k1][r1_idx]
                       if v1_val == 0: continue
                       for j2 in range(m2):
                           for k2 in range(p2):
                               v2_val = V2[j2*p2+k2][r2_idx]
                               if v2_val == 0: continue
                               j = j1*m2 + j2
                               k = k1*p2 + k2
                               V[j*p+k][r_idx] += v1_val * v2_val
       
       # W indices: k, i
       for r1_idx in range(r1):
           for r2_idx in range(r2):
               r_idx = r1_idx * r2 + r2_idx
               for k1 in range(p1):
                   for i1 in range(n1):
                       w1_val = W1[k1*n1+i1][r1_idx]
                       if w1_val == 0: continue
                       for k2 in range(p2):
                           for i2 in range(n2):
                               w2_val = W2[k2*n2+i2][r2_idx]
                               if w2_val == 0: continue
                               k = k1*p2 + k2
                               i = i1*n2 + i2
                               W[k*n+i][r_idx] += w1_val * w2_val
       
       return {'dims': (n, m, p), 'rank': r, 'U': U, 'V': V, 'W': W}
   