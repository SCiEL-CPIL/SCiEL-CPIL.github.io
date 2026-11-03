DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 27 * 7 = 189
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms definitions
    # Each term is a list of (local_row, local_col, coeff)
    # For A (U)
    A_terms = [
        [(0, 0, 1), (1, 1, 1)],  # M1
        [(1, 0, 1), (1, 1, 1)],  # M2
        [(0, 0, 1)],             # M3
        [(1, 1, 1)],             # M4
        [(0, 0, 1), (0, 1, 1)],  # M5
        [(1, 0, 1), (0, 0, -1)], # M6
        [(0, 1, 1), (1, 1, -1)]  # M7
    ]
    
    # For B (V)
    B_terms = [
        [(0, 0, 1), (1, 1, 1)],  # M1
        [(0, 0, 1)],             # M2
        [(0, 1, 1), (1, 1, -1)], # M3
        [(1, 0, 1), (0, 0, -1)], # M4
        [(1, 1, 1)],             # M5
        [(0, 0, 1), (0, 1, 1)],  # M6
        [(1, 0, 1), (1, 1, 1)]   # M7
    ]
    
    # For C (W)
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    C_terms = [
        [(0, 0, 1), (1, 1, 1)],  # M1
        [(1, 0, 1), (1, 1, -1)], # M2
        [(0, 1, 1), (1, 1, 1)],  # M3
        [(0, 0, 1), (1, 0, 1)],  # M4
        [(0, 0, -1), (0, 1, 1)], # M5
        [(1, 1, 1)],             # M6
        [(0, 0, 1)]              # M7
    ]
    
    col = 0
    for r in range(3):
        for c in range(3):
            for s in range(3):
                for t in range(7):
                    # Process A
                    for lr, lc, coeff in A_terms[t]:
                        i = 2 * r + lr
                        j = 2 * c + lc
                        U[i * m + j][col] = coeff
                        
                    # Process B
                    for lr, lc, coeff in B_terms[t]:
                        j = 2 * c + lr
                        k = 2 * s + lc
                        V[j * p + k][col] = coeff
                        
                    # Process C
                    for lr, lc, coeff in C_terms[t]:
                        k = 2 * r + lr  # Wait, C row is 2r+lr?
                        i = 2 * s + lc  # Wait, C col is 2s+lc?
                        # Let's re-verify C indices.
                        # C_{r,s} is at block row r, block col s.
                        # Global row of C_{r,s} is 2r + lr_local? No.
                        # A_{r,c} is rows 2r..2r+1.
                        # B_{c,s} is cols 2s..2s+1.
                        # Product is 2x2.
                        # The result occupies rows 2r..2r+1 and cols 2s..2s+1 in C.
                        # So global row is 2r + lr_local.
                        # Global col is 2s + lc_local.
                        # W index is k * n + i, where k is row, i is col.
                        # So k = 2r + lr, i = 2s + lc.
                        W[k * n + i][col] = coeff
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
