"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # all 6
    
    # Strassen's 7 multiplication steps for 2x2 blocks
    # U entries: (row_in_A, coeff) where row_in_A = i*6 + j
    # V entries: (row_in_B, coeff) where row_in_B = j*6 + k
    # W entries: (row_in_C, coeff) where row_in_C = k*6 + i
    
    # For block pair (a, b, c):
    # A block: rows 2a,2a+1; cols 2b,2b+1
    # B block: rows 2b,2b+1; cols 2c,2c+1
    # C block: rows 2c,2c+1; cols 2a,2a+1
    
    # Strassen steps (coefficients for each of the 4 sub-blocks):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B11)
    # M4 = A22(B22-B12)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # For each step, define which A sub-blocks and B sub-blocks are used,
    # and which C sub-blocks they contribute to with what coefficients.
    
    # Step s: list of (A_sub, coeff_A, B_sub, coeff_B) for the product,
    # and list of (C_sub, coeff_C) for where it goes.
    # Sub-blocks indexed: 11=(0,0), 12=(0,1), 21=(1,0), 22=(1,1)
    
    # A_sub index: 0=A11, 1=A12, 2=A21, 3=A22
    # B_sub index: 0=B11, 1=B12, 2=B21, 3=B22
    # C_sub index: 0=C11, 1=C12, 2=C21, 3=C22
    
    # Each step has exactly one A term and one B term (they're sums/diffs of single sub-blocks in standard Strassen)
    # Actually no - some steps have sums. Let me define it as:
    # step: (list of (A_sub, coeff), list of (B_sub, coeff), list of (C_sub, coeff))
    
    steps = [
        # M1 = (A11+A22)(B11+B22)
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),  # C11:+1, C22:+1
        # M2 = (A21+A22)B11
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),  # C21:+1, C22:-1
        # M3 = A11(B21-B11)
        ([(0, 1)], [(2, 1), (0, -1)], [(1, 1), (3, 1)]),  # C12:+1, C22:+1
        # M4 = A22(B22-B12)
        ([(3, 1)], [(3, 1), (1, -1)], [(0, 1), (2, 1)]),  # C11:+1, C21:+1
        # M5 = (A11+A12)B22
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),  # C11:-1, C12:+1
        # M6 = (A21-A11)(B11+B12)
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),  # C22:+1
        # M7 = (A12-A22)(B21+B22)
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),  # C11:+1
    ]
    
    rank = 9 * 7  # 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for step in steps:
                    A_terms, B_terms, C_terms = step
                    
                    # For each A sub-block term, add to U
                    for (sub_a, coeff_a) in A_terms:
                        # A_sub index: 0=A11(row0,col0), 1=A12(row0,col1), 2=A21(row1,col0), 3=A22(row1,col1)
                        row_offset = sub_a // 2  # 0 or 1
                        col_offset = sub_a % 2   # 0 or 1
                        i = 2 * a + row_offset
                        j = 2 * b + col_offset
                        u_idx = i * 6 + j
                        U[u_idx][col] += coeff_a
                    
                    # For each B sub-block term, add to V
                    for (sub_b, coeff_b) in B_terms:
                        # B_sub index: 0=B11(row0,col0), 1=B12(row0,col1), 2=B21(row1,col0), 3=B22(row1,col1)
                        row_offset = sub_b // 2
                        col_offset = sub_b % 2
                        j = 2 * b + row_offset
                        k = 2 * c + col_offset
                        v_idx = j * 6 + k
                        V[v_idx][col] += coeff_b
                    
                    # For each C sub-block term, add to W
                    for (sub_c, coeff_c) in C_terms:
                        # C_sub index: 0=C11(row0,col0), 1=C12(row0,col1), 2=C21(row1,col0), 3=C22(row1,col1)
                        row_offset = sub_c // 2
                        col_offset = sub_c % 2
                        k = 2 * c + row_offset
                        i = 2 * a + col_offset
                        w_idx = k * 6 + i
                        W[w_idx][col] += coeff_c
                    
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
