import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2
    # A indices: 0,1,2,3 -> a00, a01, a10, a11
    # B indices: 0,1,2,3 -> b00, b01, b10, b11
    U_strassen = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 0, -1]
    ]
    V_strassen = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    # C reconstruction: c_rc = sum_s C_strassen[rc][s] * m_s
    # c00 = m1 - m4 + m5 - m7
    # c01 = m3 + m7
    # c10 = m2 + m4
    # c11 = m1 - m2 + m5 + m6
    C_strassen = [
        [1, 0, 0, -1, 1, 0, -1],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 0, 0, 1, 1, 0]
    ]
    
    # Block definitions for 4x4
    # A blocks:
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    # Helper to get vector for a block entry
    def get_block_vec(block, r, c):
        vec = [0] * 16
        # block: 0=A00, 1=A01, 2=A10, 3=A11
        # r, c in 0,1
        # Global i = block_row * 2 + r
        # Global j = block_col * 2 + c
        br = block // 2
        bc = block % 2
        i = br * 2 + r
        j = bc * 2 + c
        vec[i * 4 + j] = 1
        return vec

    # Define the 7 block products M_l
    # Each M_l is X_l * Y_l
    # X_l and Y_l are 2x2 matrices of linear forms
    # We represent each entry of X_l and Y_l as a list of 4 vectors (for r,c in 0,1)
    # Actually, we just need the linear form for each entry.
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # We'll define X_l and Y_l as 2x2 arrays of 16-vectors
    def make_block_sum(blocks, signs):
        # blocks: list of block indices
        # signs: list of +/- 1
        vecs = [[0]*16 for _ in range(4)] # 4 entries in 2x2
        for b, s in zip(blocks, signs):
            for r in range(2):
                for c in range(2):
                    v = get_block_vec(b, r, c)
                    for idx in range(16):
                        vecs[r*2+c][idx] += s * v[idx]
        return vecs

    # X_l definitions
    X1 = make_block_sum([0, 3], [1, 1])
    X2 = make_block_sum([2, 3], [1, 1])
    X3 = make_block_sum([0], [1])
    X4 = make_block_sum([3], [1])
    X5 = make_block_sum([0, 1], [1, 1])
    X6 = make_block_sum([2, 0], [1, -1])
    X7 = make_block_sum([1, 3], [1, -1])
    
    Y1 = make_block_sum([0, 3], [1, 1])
    Y2 = make_block_sum([0], [1])
    Y3 = make_block_sum([1, 3], [1, -1])
    Y4 = make_block_sum([2, 0], [1, -1])
    Y5 = make_block_sum([3], [1])
    Y6 = make_block_sum([0, 1], [1, 1])
    Y7 = make_block_sum([2, 3], [1, 1])
    
    X_blocks = [X1, X2, X3, X4, X5, X6, X7]
    Y_blocks = [Y1, Y2, Y3, Y4, Y5, Y6, Y7]
    
    # C block coefficients
    # C00 = M1 - M4 + M5 - M7
    # C01 = M3 + M7
    # C10 = M2 + M4
    # C11 = M1 - M2 + M5 + M6
    C_block_coeffs = [
        [1, 0, 0, -1, 1, 0, -1],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 0, 0, 1, 1, 0]
    ]
    
    # Mapping from block entry (r,c) to global W row
    # Block 0 (C00): i=r, k=c. Row = 4c + r
    # Block 1 (C01): i=r, k=2+c. Row = 4(2+c) + r
    # Block 2 (C10): i=2+r, k=c. Row = 4c + 2 + r
    # Block 3 (C11): i=2+r, k=2+c. Row = 4(2+c) + 2 + r
    
    def get_w_row(block, r, c):
        if block == 0:
            return 4*c + r
        elif block == 1:
            return 4*(2+c) + r
        elif block == 2:
            return 4*c + 2 + r
        else:
            return 4*(2+c) + 2 + r

    col = 0
    for l in range(7):
        X_l = X_blocks[l]
        Y_l = Y_blocks[l]
        for s in range(7):
            u_str = U_strassen[s]
            v_str = V_strassen[s]
            
            u_vec = [0] * 16
            v_vec = [0] * 16
            
            # Compute u_vec
            for idx_a in range(4):
                if u_str[idx_a] != 0:
                    r = idx_a // 2
                    c = idx_a % 2
                    for idx in range(16):
                        u_vec[idx] += u_str[idx_a] * X_l[idx_a][idx]
                        
            # Compute v_vec
            for idx_b in range(4):
                if v_str[idx_b] != 0:
                    r = idx_b // 2
                    c = idx_b % 2
                    for idx in range(16):
                        v_vec[idx] += v_str[idx_b] * Y_l[idx_b][idx]

            # Update U and V
            for idx in range(16):
                U[idx][col] = u_vec[idx]
                V[idx][col] = v_vec[idx]
                
            # Update W
            # Contribution of this product to M_l entries
            # Then map to C blocks
            for r in range(2):
                for c in range(2):
                    # Coefficient in M_l[r,c]
                    coeff_ml = C_strassen[r*2+c][s]
                    if coeff_ml == 0:
                        continue
                    # This M_l contributes to C blocks
                    for b in range(4):
                        coeff_cb = C_block_coeffs[b][l]
                        if coeff_cb == 0:
                            continue
                        total_coeff = coeff_ml * coeff_cb
                        if total_coeff == 0:
                            continue
                        w_row = get_w_row(b, r, c)
                        W[w_row][col] += total_coeff
                        
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
