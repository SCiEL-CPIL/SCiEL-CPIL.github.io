def construct():
    n, m, p = 4, 4, 8
    
    # Strassen 2x2x2 terms (U, V, W)
    # Indices 0..3 for 2x2 matrices
    S = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]), # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]), # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]), # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]), # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]), # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0])  # M7
    ]
    
    # Block combinations for 4x4x4 (2x2 blocks of 2x2)
    # A blocks: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # B blocks: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # C blocks: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # Strassen block-level equations:
    # S1 = (A00+A11)(B00+B11) -> C00 += S1, C11 += S1
    # S2 = (A10+A11)B00      -> C10 += S2, C11 -= S2
    # S3 = A00(B01-B11)      -> C01 += S3, C11 += S3
    # S4 = A11(B10-B00)      -> C00 += S4, C10 += S4
    # S5 = (A00+A01)B11      -> C00 -= S5, C01 += S5
    # S6 = (A10-A00)(B00+B01)-> C11 += S6
    # S7 = (A01-A11)(B10+B11)-> C00 += S7
    
    # For each of the 7 block terms, we have 7 scalar Strassen terms.
    # Total 49 terms.
    
    # We need to map the block-level coefficients to the 4x4 flat vectors.
    # Block b covers rows 2*(b//2)..2*(b//2)+1 and cols 2*(b%2)..2*(b%2)+1.
    
    def get_block_coeffs(block_list):
        # Returns a dict {flat_idx: coeff}
        coeffs = {}
        for b_idx, coeff in block_list:
            bi, bj = b_idx // 2, b_idx % 2
            for li in range(2):
                for lj in range(2):
                    r = 2*bi + li
                    c = 2*bj + lj
                    flat = r*4 + c
                    coeffs[flat] = coeffs.get(flat, 0) + coeff
        return coeffs

    A_blocks = [
        [(0, 1), (3, 1)],    # S1
        [(2, 1), (3, 1)],    # S2
        [(0, 1)],            # S3
        [(3, 1)],            # S4
        [(0, 1), (1, 1)],    # S5
        [(2, 1), (0, -1)],   # S6
        [(1, 1), (3, -1)]    # S7
    ]
    B_blocks = [
        [(0, 1), (3, 1)],    # S1
        [(0, 1)],            # S2
        [(1, 1), (3, -1)],   # S3
        [(2, 1), (0, -1)],   # S4
        [(3, 1)],            # S5
        [(0, 1), (1, 1)],    # S6
        [(2, 1), (3, 1)]     # S7
    ]
    C_blocks = [
        [(0, 1), (3, 1), (4, -1), (6, 1)], # C00
        [(2, 1), (4, 1)],                  # C01
        [(1, 1), (3, 1)],                  # C10
        [(0, 1), (1, -1), (2, 1), (5, 1)]  # C11
    ]

    U_44 = [[0]*49 for _ in range(16)]
    V_44 = [[0]*49 for _ in range(16)]
    W_44 = [[0]*49 for _ in range(16)]

    for s_idx in range(7):
        A_coeffs = get_block_coeffs(A_blocks[s_idx])
        B_coeffs = get_block_coeffs(B_blocks[s_idx])
        
        for str_idx in range(7):
            t = s_idx * 7 + str_idx
            u_s, v_s, w_s = S[str_idx]
            
            # Fill U and V
            for flat_idx, coeff in A_coeffs.items():
                # flat_idx is r*4+c. r=2*bi+li, c=2*bj+lj.
                # local li = r%2, lj = c%2.
                r = flat_idx // 4
                c = flat_idx % 4
                li = r % 2
                lj = c % 2
                U_44[flat_idx][t] += coeff * u_s[li*2 + lj]
                
            for flat_idx, coeff in B_coeffs.items():
                r = flat_idx // 4
                c = flat_idx % 4
                li = r % 2
                lj = c % 2
                V_44[flat_idx][t] += coeff * v_s[li*2 + lj]

            # Fill W
            # W_44 index is r*4+c for output C.
            # We need to add contributions from each C_block that involves this s_idx.
            for c_idx in range(4):
                for (s_ref, c_coeff) in C_blocks[c_idx]:
                    if s_ref == s_idx:
                        # c_idx is the block index in C.
                        # Block c_idx covers rows 2*(c_idx//2)..+1 and cols 2*(c_idx%2)..+1.
                        bi, bj = c_idx // 2, c_idx % 2
                        for li in range(2):
                            for lj in range(2):
                                r = 2*bi + li
                                c = 2*bj + lj
                                flat_idx = r*4 + c
                                W_44[flat_idx][t] += c_coeff * w_s[li*2 + lj]
                        break # Only one s_ref matches s_idx in the list for a given c_idx? 
                             # Actually, C_blocks[c_idx] lists all s_idx contributing to C_block c_idx.
                             # So for a fixed s_idx, we check if it's in the list.
                             # Yes, `if s_ref == s_idx` handles it.

    # Construct 4x4x8
    rank = 98
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]

    # U is same for both halves
    for i in range(16):
        for t in range(49):
            U[i][t] = U_44[i][t]
            U[i][49+t] = U_44[i][t]

    # V
    for i in range(4):
        for j in range(8):
            for t in range(49):
                if j < 4:
                    V[i*8 + j][t] = V_44[i*4 + j][t]
                else:
                    V[i*8 + j][49+t] = V_44[i*4 + (j-4)][t]

    # W
    for i in range(4):
        for k in range(8):
            for t in range(49):
                if k < 4:
                    W[k*4 + i][t] = W_44[i*4 + k][t]
                else:
                    W[k*4 + i][49+t] = W_44[i*4 + (k-4)][t]

    return {"U": U, "V": V, "W": W}
