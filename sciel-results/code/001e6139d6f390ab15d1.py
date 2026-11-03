"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    b = 3  # Block size
    
    rank = 7 * 27  # 7 Strassen products * 27 scalar products per 3x3 mult
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen definitions
    # A_combos[k]: list of (block_idx, sign) for the first factor of M_k
    # B_combos[k]: list of (block_idx, sign) for the second factor of M_k
    # C_map[k]: list of (block_idx, sign) for the contribution of M_k to C
    
    A_combos = [
        [(0, 1), (3, 1)],  # M1: A00 + A11
        [(2, 1), (3, 1)],  # M2: A10 + A11
        [(0, 1)],          # M3: A00
        [(3, 1)],          # M4: A11
        [(0, 1), (1, 1)],  # M5: A00 + A01
        [(2, 1), (0, -1)], # M6: A10 - A00
        [(1, 1), (3, -1)]  # M7: A01 - A11
    ]
    
    B_combos = [
        [(0, 1), (3, 1)],  # M1: B00 + B11
        [(0, 1)],          # M2: B00
        [(1, 1), (3, -1)], # M3: B01 - B11
        [(2, 1), (0, -1)], # M4: B10 - B00
        [(3, 1)],          # M5: B11
        [(0, 1), (1, 1)],  # M6: B00 + B01
        [(2, 1), (3, 1)]   # M7: B10 + B11
    ]
    
    # C contributions: (C_block, sign)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    C_map = [
        [(0, 1), (3, 1)],  # M1
        [(2, 1), (3, -1)], # M2
        [(1, 1), (3, 1)],  # M3
        [(0, 1), (2, 1)],  # M4
        [(0, -1), (1, 1)], # M5
        [(3, 1)],          # M6
        [(0, 1)]           # M7
    ]
    
    # Generate decomposition
    for k in range(7):
        a_combos = A_combos[k]
        b_combos = B_combos[k]
        c_map = C_map[k]
        
        # Iterate over the 27 scalar products of a 3x3 multiplication
        for r in range(b):
            for c in range(b):
                for s in range(b):
                    # This corresponds to the term A_{rs} * B_{sc} in the 3x3 product
                    # We need to map A_{rs} of the combined matrix to original A elements
                    # and B_{sc} of the combined matrix to original B elements
                    
                    # Update U
                    for a_block, a_sign in a_combos:
                        a_i = a_block // 2
                        a_j = a_block % 2
                        a_row = a_i * b + r
                        a_col = a_j * b + s
                        a_idx = a_row * m + a_col
                        U[a_idx][col] += a_sign
                    
                    # Update V
                    for b_block, b_sign in b_combos:
                        b_i = b_block // 2
                        b_j = b_block % 2
                        b_row = b_i * b + s
                        b_col = b_j * b + c
                        b_idx = b_row * p + b_col
                        V[b_idx][col] += b_sign
                    
                    # Update W
                    for c_block, c_sign in c_map:
                        c_i = c_block // 2
                        c_j = c_block % 2
                        c_row = c_i * b + r
                        c_col = c_j * b + c
                        # W index convention: k * n + i
                        # Here c_col is the column of C (k), c_row is the row of C (i)
                        w_idx = c_col * n + c_row
                        W[w_idx][col] += c_sign
                    
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
