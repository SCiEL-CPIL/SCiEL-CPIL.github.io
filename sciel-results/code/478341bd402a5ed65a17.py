   def _strassen_4x4_4x4_factors():
       I2 = np.array([[1,0],[0,1]])
       A00=np.array([[1,0],[0,0]]); A01=np.array([[0,1],[0,0]])
       A10=np.array([[0,0],[1,0]]); A11=np.array([[0,0],[0,1]])
       B00=np.array([[1,0],[0,0]]); B01=np.array([[0,1],[0,0]])
       B10=np.array([[0,0],[1,0]]); B11=np.array([[0,0],[0,1]])
       # (U2, V2) for each Strassen product; W2 = I2 (result is 2x2 matrix)
       terms2 = [
           (A00+A11, B00+B11),  # M1
           (A10+A11, B00),      # M2
           (A00, B10-B11),      # M3
           (A11, B01-B10),      # M4
           (A00+A01, B11),      # M5
           (A10-A00, B00+B10),  # M6
           (A01-A11, B01+B11),  # M7
       ]
       comb = {
           (0,0): [(0,1),(3,1),(4,-1),(6,1)],
           (0,1): [(2,1),(4,1)],
           (1,0): [(1,1),(3,1)],
           (1,1): [(0,1),(1,-1),(2,1),(6,1)],
       }
       U4s,V4s,W4s = [],[],[]
       for t in range(7):
           U2,V2 = terms2[t]
           for (ci,cj), parts in comb.items():
               coeff = sum(s for (idx,s) in parts if idx==t)
               if coeff==0: continue
               for c in range(2):  # block-internal matmul index
                   U4=np.zeros((4,4)); V4=np.zeros((4,4)); W4=np.zeros((4,4))
                   for a in range(2):
                       for b in range(2):
                           # U4: A-input at block (ci,cj), pos (a,c)
                           U4[2*ci+a, 2*cj+c] = U2[a,c]
                           # V4: B-input at block (ci,cj), pos (c,b)
                           V4[2*ci+c, 2*cj+b] = V2[c,b]
                           # W4: C-output at block (ci,cj), pos (a,b)
                           W4[2*ci+a, 2*cj+b] = coeff
                   U4s.append(U4); V4s.append(V4); W4s.append(W4)
       ...
   