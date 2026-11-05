def get_bclr_3x3():
    terms = []
    # A, B, C are 3x3.
    # Indices:
    # A: 0..8
    # B: 0..8
    # C: 0..8 (col-major? W[j*3+i])
    
    # 1. A00 B00 -> C00
    terms.append(([(0,1)], [(0,1)], [(0,1)]))
    # 2. A01 B10 -> C00
    terms.append(([(1,1)], [(3,1)], [(0,1)]))
    # 3. A02 B20 -> C00
    terms.append(([(2,1)], [(6,1)], [(0,1)]))
    # 4. A10 B01 -> C10 (idx 1)
    terms.append(([(3,1)], [(1,1)], [(1,1)]))
    # 5. A11 B11 -> C10 (idx 1) AND C11 (idx 4)? No.
    # The BCLR decomposition is:
    # P1 = A00 B00
    # P2 = A01 B10
    # P3 = A02 B20
    # P4 = A10 B01
    # P5 = A11 B11
    # P6 = A12 B21
    # P7 = A20 B02
    # P8 = A21 B12
    # P9 = A22 B22
    # P10 = (A00+A11) (B01+B10) ... No.
