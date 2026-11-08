import time

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a linear combination of blocks to U, V, W
    # block is (r0, c0) for A, B, C
    # A index: r0*2 + dr, c0*2 + dc
    # B index: r0*2 + dr, c0*2 + dc
    # C index: (c0*2 + dc)*n + (r0*2 + dr)
    
    # Strassen 2x2 scalar formulas
    # p1 = (x11 + x22)(y11 + y22)
    # p2 = (x21 + x22)y11
    # p3 = x11(y12 - y22)
    # p4 = x22(y21 - y11)
    # p5 = (x11 + x12)y22
    # p6 = (x21 - x11)(y11 + y12)
    # p7 = (x12 - x22)(y21 + y22)
    
    # x entries: x11 (0,0), x12 (0,1), x21 (1,0), x22 (1,1)
    # y entries: y11 (0,0), y12 (0,1), y21 (1,0), y22 (1,1)
    
    strassen_terms = [
        # (x_coeffs, y_coeffs)
        # x_coeffs: dict (dr, dc) -> coeff
        # y_coeffs: dict (dr, dc) -> coeff
        [ { (0,0): 1, (1,1): 1 }, { (0,0): 1, (1,1): 1 } ], # p1
        [ { (1,0): 1, (1,1): 1 }, { (0,0): 1 } ],           # p2
        [ { (0,0): 1 }, { (0,1): 1, (1,1): -1 } ],          # p3
        [ { (1,1): 1 }, { (1,0): 1, (0,0): -1 } ],          # p4
        [ { (0,0): 1, (0,1): 1 }, { (1,1): 1 } ],           # p5
        [ { (1,0): 1, (0,0): -1 }, { (0,0): 1, (0,1): 1 } ],# p6
        [ { (0,1): 1, (1,1): -1 }, { (1,0): 1, (1,1): 1 } ] # p7
    ]
    
    # Block expressions for M_b = X_b Y_b
    # X_b is a linear combo of A blocks (r0, c0)
    # Y_b is a linear combo of B blocks (r0, c0)
    block_exprs = [
        # b=0: M0 = (A00 + A11)(B00 + B11)
        [ { (0,0): 1, (1,1): 1 }, { (0,0): 1, (1,1): 1 } ],
        # b=1: M1 = (A10 + A11)B00
        [ { (1,0): 1, (1,1): 1 }, { (0,0): 1 } ],
        # b=2: M2 = A00(B01 - B11)
        [ { (0,0): 1 }, { (0,1): 1, (1,1): -1 } ],
        # b=3: M3 = A11(B10 - B00)
        [ { (1,1): 1 }, { (1,0): 1, (0,0): -1 } ],
        # b=4: M4 = (A00 + A01)B11
        [ { (0,0): 1, (0,1): 1 }, { (1,1): 1 } ],
        # b=5: M5 = (A10 - A00)(B00 + B01)
        [ { (1,0): 1, (0,0): -1 }, { (0,0): 1, (0,1): 1 } ],
        # b=6: M6 = (A01 - A11)(B10 + B11)
        [ { (0,1): 1, (1,1): -1 }, { (1,0): 1, (1,1): 1 } ]
    ]
    
    # C block combinations
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    c_block_coeffs = {
        (0,0): { 0: 1, 3: 1, 4: -1, 6: 1 },
        (0,1): { 2: 1, 4: 1 },
        (1,0): { 1: 1, 3: 1 },
        (1,1): { 0: 1, 1: -1, 2: 1, 5: 1 }
    }
    
    # Strassen output coefficients
    # z11 = p1 + p4 - p5 + p7
    # z12 = p3 + p5
    # z21 = p2 + p4
    # z22 = p1 - p2 + p3 + p6
    # p indices 0..6 correspond to s=0..6
    z_coeffs = {
        (0,0): { 0: 1, 3: 1, 4: -1, 6: 1 },
        (0,1): { 2: 1, 4: 1 },
        (1,0): { 1: 1, 3: 1 },
        (1,1): { 0: 1, 1: -1, 2: 1, 5: 1 }
    }
    
    for b in range(7):
        x_blocks = block_exprs[b][0]
        y_blocks = block_exprs[b][1]
        
        for s in range(7):
            c = b * 7 + s
            if time.time() > deadline:
                break
                
            x_coeffs = strassen_terms[s][0]
            y_coeffs = strassen_terms[s][1]
            
            # U: A entries
            for (br, bc), coeff_x_block in x_blocks.items():
                for (dr, dc), coeff_x_entry in x_coeffs.items():
                    coeff = coeff_x_block * coeff_x_entry
                    if coeff != 0:
                        r = br * 2 + dr
                        col = bc * 2 + dc
                        U[r * m + col][c] += coeff
                        
            # V: B entries
            for (br, bc), coeff_y_block in y_blocks.items():
                for (dr, dc), coeff_y_entry in y_coeffs.items():
                    coeff = coeff_y_block * coeff_y_entry
                    if coeff != 0:
                        r = br * 2 + dr
                        col = bc * 2 + dc
                        V[r * p + col][c] += coeff
                        
            # W: C entries
            for (cr, cc), coeff_m in c_block_coeffs.items():
                if b not in coeff_m:
                    continue
                coeff_m_b = coeff_m[b]
                if coeff_m_b == 0:
                    continue
                for (dr, dc), coeff_z in z_coeffs.items():
                    if s not in coeff_z:
                        continue
                    coeff_z_s = coeff_z[s]
                    if coeff_z_s == 0:
                        continue
                    coeff = coeff_m_b * coeff_z_s
                    if coeff != 0:
                        r = cr * 2 + dr
                        col = cc * 2 + dc
                        W[r * n + col][c] += coeff
                        
    return {"U": U, "V": V, "W": W}
