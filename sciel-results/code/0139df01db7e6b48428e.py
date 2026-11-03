def construct():
    n, m, p = 2, 4, 4
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # Local indices: a00, a01, a10, a11 (row, col)
    # b00, b01, b10, b11 (row, col)
    # c00, c01, c10, c11 (row, col)
    
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b11+b01)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We define the 7 patterns for U, V, W for a single 2x2 block
    # U is 2x2 local, V is 2x2 local, W is 2x2 local
    
    strassen_U = [
        [1, 0, 0, 1], # M1: a00, a11
        [0, 0, 1, 1], # M2: a10, a11
        [1, 0, 0, 0], # M3: a00
        [0, 0, 0, 1], # M4: a11
        [1, 1, 0, 0], # M5: a00, a01
        [1, 0, -1, 0],# M6: a10, a00 (a10 - a00) -> wait, M6 is (a10-a00). So U has 1 at a10, -1 at a00.
        [0, 1, 0, -1] # M7: a01, a11 (a01 - a11) -> U has 1 at a01, -1 at a11.
    ]
    
    strassen_V = [
        [1, 0, 0, 1], # M1: b00, b11
        [1, 0, 0, 0], # M2: b00
        [0, -1, 1, 0],# M3: b10, b11 (b10 - b11) -> V has 1 at b10, -1 at b11.
        [0, 1, -1, 0],# M4: b01, b00 (b01 - b00) -> V has 1 at b01, -1 at b00.
        [0, 0, 0, 1], # M5: b11
        [1, 0, 1, 0], # M6: b00, b10 (b00 + b10)
        [0, 1, 0, 1]  # M7: b01, b11 (b11 + b01)
    ]
    
    strassen_W = [
        [1, 0, 0, -1], # M1: c00, c11 (c00 + c11? No, C00=M1, C11=M1. So W has 1 at c00, 1 at c11)
        # Let's re-verify C formulas.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # W[r] is the vector of coefficients for C.
        # W[0] (for M1): 1 at C00, 1 at C11.
        # W[1] (for M2): 1 at C10, -1 at C11.
        # W[2] (for M3): 1 at C01, 1 at C11.
        # W[3] (for M4): 1 at C00, 1 at C10.
        # W[4] (for M5): -1 at C00, 1 at C01.
        # W[5] (for M6): 1 at C11.
        # W[6] (for M7): 1 at C00.
        
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1],# M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Define the 4 blocks
    # Block 0: i in {0,1}, j in {0,1}, k in {0,1}
    # Block 1: i in {0,1}, j in {2,3}, k in {0,1}
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3}
    # Block 3: i in {0,1}, j in {2,3}, k in {2,3}
    
    blocks = [
        (0, 0, 0), # i_offset, j_offset, k_offset
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2)
    ]
    
    col = 0
    for i_off, j_off, k_off in blocks:
        for s in range(7):
            # Fill U
            for li in range(2):
                for lj in range(2):
                    val = strassen_U[s][li*2 + lj]
                    if val != 0:
                        gi = i_off + li
                        gj = j_off + lj
                        U[gi * m + gj][col] += val
                        
            # Fill V
            for lj in range(2):
                for lk in range(2):
                    val = strassen_V[s][lj*2 + lk]
                    if val != 0:
                        gj = j_off + lj
                        gk = k_off + lk
                        V[gj * p + gk][col] += val
                        
            # Fill W
            for lk in range(2):
                for li in range(2):
                    val = strassen_W[s][lk*2 + li] # Note: W is indexed by k, i?
                    # Task says W[k*n+i, r].
                    # Local c is (li, lk) where li is row (i), lk is col (k).
                    # So global index is gk * n + gi.
                    if val != 0:
                        gi = i_off + li
                        gk = k_off + lk
                        W[gk * n + gi][col] += val
                        
            col += 1
            
    return {"U": U, "V": V, "W": W}
