def construct():
    n, m, p = 4, 4, 7
    rank = 97  # 49 (Strassen for 4x4x4) + 48 (schoolbook for 4x4x3)
    
    U = [[0] * rank for _ in range(n * m)]  # 16 x 97
    V = [[0] * rank for _ in range(m * p)]  # 28 x 97
    W = [[0] * rank for _ in range(p * n)]  # 28 x 97
    
    # Strassen-2x2 tensor decomposition
    # Each term: (u, v, w) where u,v,w are length-4 vectors
    # A indexed as A[i*2+j], B as B[j*2+k], C as C[k*2+i]
    strassen_2x2 = [
        ([1,0,0,1], [1,0,0,1], [1,1,1,-1]),   # M1
        ([0,1,1,0], [1,0,0,0], [0,0,1,0]),     # M2
        ([1,0,0,0], [0,1,0,-1], [0,1,0,0]),    # M3
        ([0,0,0,1], [0,-1,1,0], [-1,0,1,0]),   # M4
        ([1,1,0,0], [0,0,0,1], [1,0,0,0]),     # M5
        ([-1,1,0,0], [1,1,0,0], [0,0,0,-1]),   # M6
        ([0,1,-1,0], [0,1,1,0], [-1,0,0,-1]),  # M7
    ]
    
    # Top-level coefficients for 4x4x4 Strassen
    # Block index: 0=00, 1=01, 2=10, 3=11
    top_A_coeffs = [
        {0: 1, 3: 1},       # M1: A00 + A11
        {2: 1, 3: 1},       # M2: A10 + A11
        {0: 1},             # M3: A00
        {3: 1},             # M4: A11
        {0: 1, 1: 1},       # M5: A00 + A01
        {2: 1, 0: -1},      # M6: A10 - A00
        {1: 1, 3: -1},      # M7: A01 - A11
    ]
    
    top_B_coeffs = [
        {0: 1, 3: 1},       # M1: B00 + B11
        {0: 1},             # M2: B00
        {1: 1, 3: -1},      # M3: B01 - B11
        {2: 1, 0: -1},      # M4: B10 - B00
        {3: 1},             # M5: B11
        {0: 1, 1: 1},       # M6: B00 + B01
        {2: 1, 3: 1},       # M7: B10 + B11
    ]
    
    top_C_coeffs = [
        {0: 1, 1: 1, 2: 1, 3: -1},   # M1
        {2: 1},                         # M2
        {1: 1},                         # M3
        {0: -1, 2: 1},                  # M4
        {0: 1},                         # M5
        {3: -1},                        # M6
        {0: -1, 1: 1, 3: -1},          # M7
    ]
    
    def A_block_entries(b):
        bi, bj = b // 2, b % 2
        return [(2*bi + i)*4 + (2*bj + j) for i in range(2) for j in range(2)]
    
    def B_block_entries(b):
        bi, bj = b // 2, b % 2
        return [(2*bi + i)*4 + (2*bj + j) for i in range(2) for j in range(2)]
    
    def C_block_entries(b):
        bi, bj = b // 2, b % 2
        return [(2*bi + k)*4 + (2*bj + i) for k in range(2) for i in range(2)]
    
    # Fill in the Strassen part (columns 0-48)
    for r in range(7):
        for s in range(7):
            col = r * 7 + s
            u_sub, v_sub, w_sub = strassen_2x2[s]
            
            for blk, coeff in top_A_coeffs[r].items():
                entries = A_block_entries(blk)
                for idx, sub_idx in enumerate(entries):
                    U[sub_idx][col] += coeff * u_sub[idx]
            
            for blk, coeff in top_B_coeffs[r].items():
                entries = B_block_entries(blk)
                for idx, sub_idx in enumerate(entries):
                    V[sub_idx][col] += coeff * v_sub[idx]
            
            for blk, coeff in top_C_coeffs[r].items():
                entries = C_block_entries(blk)
                for idx, sub_idx in enumerate(entries):
                    W[sub_idx][col] += coeff * w_sub[idx]
    
    # Schoolbook part for the remaining 3 columns (columns 49-96)
    for i in range(4):
        for j in range(4):
            for k in range(3):
                col = 49 + (i * 4 + j) * 3 + k
                U[i * 4 + j][col] = 1
                V[j * 7 + (4 + k)][col] = 1
                W[(4 + k) * 4 + i][col] = 1
    
    return {"U": U, "V": V, "W": W}
