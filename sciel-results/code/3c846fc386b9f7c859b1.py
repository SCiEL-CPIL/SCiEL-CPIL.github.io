"""Program for 3x3x8 matrix multiplication tensor decomposition."""

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 3, 8
    
    rank1_terms = []
    
    def add_term(u_dict, v_dict, w_dict):
        rank1_terms.append((u_dict, v_dict, w_dict))
    
    # Block 1: A22 * B28 (2x2 * 2x8 -> 2x8)
    # Use 4 Strassen 2x2x2 blocks for 4 column pairs
    for col_group in range(4):
        k1 = col_group * 2
        k2 = col_group * 2 + 1
        
        # Global flat indices
        a00, a01, a10, a11 = 0, 1, 3, 4
        b0k1, b0k2 = k1, k2
        b1k1, b1k2 = 8 + k1, 8 + k2
        c0k1, c0k2 = k1 * 3 + 0, k1 * 3 + 1
        c1k1, c1k2 = k2 * 3 + 0, k2 * 3 + 1
        
        # Strassen 2x2x2 multipliers
        # m1 = (A00+A11)*(B0k1+B1k1)
        add_term({a00: 1, a11: 1}, {b0k1: 1, b1k1: 1}, {c0k1: 1, c1k2: 1})
        
        # m2 = (A10+A11)*B0k1
        add_term({a10: 1, a11: 1}, {b0k1: 1}, {c1k1: 1})
        
        # m3 = A00*(B0k2-B1k2)
        add_term({a00: 1}, {b0k2: 1, b1k2: -1}, {c0k2: 1})
        
        # m4 = A11*(B1k1-B0k1)
        add_term({a11: 1}, {b1k1: 1, b0k1: -1}, {c0k1: 1, c1k1: 1})
        
        # m5 = (A00+A01)*B1k2
        add_term({a00: 1, a01: 1}, {b1k2: 1}, {c0k1: -1, c0k2: 1})
        
        # m6 = (A10-A00)*(B0k1+B0k2)
        add_term({a10: 1, a00: -1}, {b0k1: 1, b0k2: 1}, {c1k2: 1})
        
        # m7 = (A01-A11)*(B1k1+B1k2)
        add_term({a01: 1, a11: -1}, {b1k1: 1, b1k2: 1}, {c0k1: 1, c1k2: 1})
    
    # Block 2: A21 * B18 (2x1 * 1x8 -> 2x8)
    # A21: rows 0,1; col 2 -> flat: 2, 5
    # B18: row 2; cols 0..7 -> flat: 16..23
    # C28: rows 0,1; cols 0..7 -> flat: k*3+0, k*3+1
    for k in range(8):
        add_term({2: 1}, {16 + k: 1}, {k * 3 + 0: 1})
        add_term({5: 1}, {16 + k: 1}, {k * 3 + 1: 1})
    
    # Block 3: A12 * B28 (1x2 * 2x8 -> 1x8)
    # A12: row 2; cols 0,1 -> flat: 6, 7
    # B28: rows 0,1; cols 0..7 -> flat: 0..15
    # C18: row 2; cols 0..7 -> flat: k*3+2
    for k in range(8):
        add_term({6: 1}, {k: 1}, {k * 3 + 2: 1})
        add_term({7: 1}, {8 + k: 1}, {k * 3 + 2: 1})
    
    # Block 4: A11 * B18 (1x1 * 1x8 -> 1x8)
    # A11: row 2, col 2 -> flat: 8
    # B18: row 2; cols 0..7 -> flat: 16..23
    # C18: row 2; cols 0..7 -> flat: k*3+2
    for k in range(8):
        add_term({8: 1}, {16 + k: 1}, {k * 3 + 2: 1})
    
    # Build U, V, W matrices
    rank = len(rank1_terms)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_dict, v_dict, w_dict) in enumerate(rank1_terms):
        for idx, val in u_dict.items():
            U[idx][r] += val
        for idx, val in v_dict.items():
            V[idx][r] += val
        for idx, val in w_dict.items():
            W[idx][r] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
