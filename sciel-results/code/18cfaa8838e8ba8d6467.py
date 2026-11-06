   def construct():
       p = 17
       # Calculate NQR
       nqr = set()
       for i in range(1, p):
           if pow(i, (p-1)//2, p) != 1:
               nqr.add(i)
       # nqr should be {3, 5, 6, 7, 10, 11, 12, 14}
       
       # Generate 9 basis vectors
       basis = []
       for shift in range(9):
           vec = 0
           for res in nqr:
               pos = (shift + res) % p
               vec |= (1 << pos)
           basis.append(vec)
           
       # Generate all 512 codewords
       words = []
       for mask in range(512):
           w = 0
           for i in range(9):
               if (mask >> i) & 1:
                   w ^= basis[i]
           # Extend
           parity = w.bit_count() % 2
           extended_w = w | (parity << 17)
           words.append(extended_w)
           
       return words
   