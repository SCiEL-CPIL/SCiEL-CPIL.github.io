"""Matrix multiplication 3x4x7 with Strassen-assisted rank 74 decomposition."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 7
    u_dict = {}
    v_dict = {}
    w_dict = {}
    
    def add_term2(u_rows, v_rows, w_rows):
        rank = len(u_dict)
        for r, val in u_rows.items():
            u_dict.setdefault(r, []).append((rank, val))
        for r, val in v_rows.items():
            v_dict.setdefault(r, []).append((rank, val))
        for r, val in w_rows.items():
            w_dict.setdefault(r, []).append((rank, val))
    
    def add_strassen(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        """Strassen 2x2x2 at A[a_r0:a_r0+2, a_c0:a_c0+2] * B[b_r0:b_r0+2, b_c0:b_c0+2] -> C[c_r0:c_r0+2, c_c0:c_c0+2]
        W index: k*n + i where k is col, i is row of C.
        """
        aU00 = a_r0 * m + a_c0
        aU01 = a_r0 * m + a_c0 + 1
        aU10 = (a_r0+1) * m + a_c0
        aU11 = (a_r0+1) * m + a_c0 + 1
        
        bV00 = b_r0 * p + b_c0
        bV01 = b_r0 * p + b_c0 + 1
        bV10 = (b_r0+1) * p + b_c0
        bV11 = (b_r0+1) * p + b_c0 + 1
        
        cW00 = c_c0 * n + c_r0       # C[0,0] -> W[col0*n + row0]
        cW01 = (c_c0+1) * n + c_r0   # C[0,1]
        cW10 = c_c0 * n + c_r0 + 1   # C[1,0]
        cW11 = (c_c0+1) * n + c_r0 + 1  # C[1,1]
        
        # M1 = (A00+A11)(B00+B11): C00+=M1, C11+=M1
        add_term2({aU00: 1, aU11: 1}, {bV00: 1, bV11: 1}, {cW00: 1, cW11: 1})
        # M2 = (A10+A11)B00: C10+=M2, C11-=M2
        add_term2({aU10: 1, aU11: 1}, {bV00: 1}, {cW10: 1, cW11: -1})
        # M3 = A00(B10-B11): C01+=M3, C11+=M3
        add_term2({aU00: 1}, {bV10: 1, bV11: -1}, {cW01: 1, cW11: 1})
        # M4 = A11(B01-B10): C00+=M4, C10+=M4
        add_term2({aU11: 1}, {bV01: 1, bV10: -1}, {cW00: 1, cW10: 1})
        # M5 = (A00+A01)B11: C01+=M5, C00-=M5
        add_term2({aU00: 1, aU01: 1}, {bV11: 1}, {cW01: 1, cW00: -1})
        # M6 = (A10-A00)(B00+B10): C11+=M6
        add_term2({aU10: 1, aU00: -1}, {bV00: 1, bV10: 1}, {cW11: 1})
        # M7 = (A01-A11)(B01+B11): C00+=M7, C01+=M7
        add_term2({aU01: 1, aU11: -1}, {bV01: 1, bV11: 1}, {cW00: 1, cW01: 1})
    
    def add_schoolbook_block(a_r_start, a_c_start, a_h, a_w, 
                             b_c_start, b_h, b_w,
                             c_r_start, c_c_start):
        for dr in range(a_h):
            for dc in range(a_w):
                for dk in range(b_w):
                    aU = (a_r_start + dr) * m + (a_c_start + dc)
                    bV = (a_c_start + dc) * p + (b_c_start + dk)
                    cW = (c_c_start + dk) * n + (c_r_start + dr)
                    add_term2({aU: 1}, {bV: 1}, {cW: 1})
    
    # P1: 2x4 x 4x4 -> 2x4 (rank 28)
    add_strassen(0, 0, 0, 0, 0, 0)
    add_strassen(0, 0, 0, 2, 0, 2)
    add_strassen(0, 2, 2, 0, 0, 0)
    add_strassen(0, 2, 2, 2, 0, 2)
    
    # P2: 2x4 x 4x3 -> 2x3 (rank 18)
    add_strassen(0, 0, 0, 4, 0, 4)
    add_schoolbook_block(0, 0, 2, 2, 6, 2, 1, 0, 6)
    add_strassen(0, 2, 2, 4, 0, 4)
    add_schoolbook_block(0, 2, 2, 2, 6, 2, 1, 0, 6)
    
    # P3: 1x4 x 4x4 -> 1x4 (rank 16)
    add_schoolbook_block(2, 0, 1, 4, 0, 4, 4, 2, 0)
    
    # P4: 1x4 x 4x3 -> 1x3 (rank 12)
    add_schoolbook_block(2, 0, 1, 4, 4, 4, 3, 2, 4)
    
    rank = len(u_dict)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, entries in u_dict.items():
        for c, val in entries:
            U[r][c] = val
    for r, entries in v_dict.items():
        for c, val in entries:
            V[r][c] = val
    for r, entries in w_dict.items():
        for c, val in entries:
            W[r][c] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
