"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Decomposition of 4x4 * 4x6 using block structure 2x2 blocks.
    # Rank = Rank(M_2,2,3) * Rank(M_2,2,2) = 11 * 7 = 77.
    
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2 * 2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = -M1 + M3 + M5 + M7
    # C10 = M2 - M3 - M6 + M1
    # C11 = -M2 + M3 + M6 - M4
    
    strassen_A = [
        [(1,0,0,0), (0,0,1,0), (1,0,0,0), (0,0,1,0)], # M1: A00+A11 (flattened 2x2)
        # Actually, easier to define coefficients for each A_ij and B_ij
    ]
    
    # Let's define the 7 Strassen multiplications explicitly.
    # Each multiplication is of the form sum(c_ik A_ik) * sum(d_jk B_jk)
    # For 2x2 matrices, indices are (0,0), (0,1), (1,0), (1,1) -> 0,1,2,3
    
    # M1: (A00+A11) * (B00+B11)
    # M2: (A10+A11) * B00
    # M3: A00 * (B10-B11)
    # M4: A11 * (B01-B00)
    # M5: (A00+A01) * B11
    # M6: (A10-A00) * (B00+B01)
    # M7: (A01-A11) * (B10+B11)
    
    # Coefficients for A and B for each of the 7 products
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    sA = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    sB = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 1, -1],# M3
        [0, 1, 0, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Output combinations for C (2x2)
    # C00, C01, C10, C11
    sC = [
        [1, 0, 0, 0, 0, 0, 1],  # M1 + M4 - M5 + M7 ? Wait.
        # M1=1, M2=2, M3=3, M4=4, M5=5, M6=6, M7=7
        # C00 = M1 + M4 - M5 + M7 -> 1, 0, 0, 1, -1, 0, 1
        # C01 = -M1 + M3 + M5 + M7 -> -1, 0, 1, 0, 1, 0, 1
        # C10 = M2 - M3 - M6 + M1 -> 1, 1, -1, 0, 0, -1, 0
        # C11 = -M2 + M3 + M6 - M4 -> 0, -1, 1, -1, 0, 1, 0
        [1, 0, 0, 1, -1, 0, 1],
        [-1, 0, 1, 0, 1, 0, 1],
        [1, 1, -1, 0, 0, -1, 0],
        [0, -1, 1, -1, 0, 1, 0]
    ]
    
    # Decomposition of 2x2 * 2x3 (Block level)
    # We split B into B_left (2x2) and B_right (2x1)
    # B_left: cols 0,1. B_right: col 2.
    # Part 1: 2x2 * 2x2 (Rank 7)
    # Part 2: 2x2 * 2x1 (Rank 4)
    
    # Block indices:
    # A: 00, 01, 10, 11 (2x2 blocks)
    # B: 00, 01, 02, 10, 11, 12 (2x2 blocks)
    # C: 00, 01, 02, 10, 11, 12 (2x2 blocks)
    
    # The 11 terms at block level:
    # Terms 1-7: Strassen for A * B_left
    # B_left uses B00, B01, B10, B11.
    # Terms 8-11: Schoolbook for A * B_right
    # B_right uses B02, B12.
    # C_right = A00*B02 + A01*B12 (for row 0)
    #           A10*B02 + A11*B12 (for row 1)
    
    # We construct U, V, W for the whole 4x4 * 4x6 problem.
    # U: (16, 77)
    # V: (24, 77)
    # W: (24, 77)
    
    rank = 77
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(U_row_indices, V_row_indices, W_row_indices, coeff_U, coeff_V, coeff_W):
        nonlocal col
        for i in range(len(U_row_indices)):
            U[U_row_indices[i]][col] += coeff_U[i]
            V[V_row_indices[i]][col] += coeff_V[i]
            W[W_row_indices[i]][col] += coeff_W[i]
        col += 1

    # Map block indices to flat indices
    # A is 4x4. Blocks are 2x2.
    # A_block(i,j) is at rows 2i..2i+1, cols 2j..2j+1.
    # Flat index for A_entry(r,c) is r*4+c.
    # A_block(0,0) entries: (0,0)->0, (0,1)->1, (1,0)->4, (1,1)->5.
    
    def get_A_block_indices(bi, bj):
        r0, c0 = 2*bi, 2*bj
        return [r0*4+c0, r0*4+c0+1, (r0+1)*4+c0, (r0+1)*4+c0+1]
        
    def get_B_block_indices(bi, bj):
        # B is 4x6. Blocks 2x2.
        # Bi is 0,1. Bj is 0,1,2.
        r0, c0 = 2*bi, 2*bj
        return [r0*6+c0, r0*6+c0+1, (r0+1)*6+c0, (r0+1)*6+c0+1]
        
    def get_C_block_indices(bi, bj):
        # C is 6x4? No, W is (p*n, rank).
        # Output is 4x6? No, A(4x4) * B(4x6) = C(4x6).
        # Trace convention: W[k*n+i, r].
        # n=4, p=6.
        # C is 4x6.
        # W index: k*n+i. k is col of C (0..5), i is row of C (0..3).
        # C_block(bi, bj) is rows 2bi..2bi+1, cols 2bj..2bj+1.
        # Entries:
        # (2bi, 2bj) -> k=2bj, i=2bi -> 2bj*4 + 2bi
        # (2bi, 2bj+1) -> k=2bj+1, i=2bi -> (2bj+1)*4 + 2bi
        # (2bi+1, 2bj) -> k=2bj, i=2bi+1 -> 2bj*4 + 2bi+1
        # (2bi+1, 2bj+1) -> k=2bj+1, i=2bi+1 -> (2bj+1)*4 + 2bi+1
        r0, c0 = 2*bi, 2*bj
        return [c0*4+r0, c0*4+r0+1, (c0+1)*4+r0, (c0+1)*4+r0+1]

    # 1. Strassen part for B_left (B00, B01, B10, B11)
    # For each of the 7 Strassen terms:
    for s in range(7):
        # Determine which A blocks are involved
        # sA[s] gives coeffs for A00, A01, A10, A11
        # A00 is block(0,0), A01 is block(0,1), A10 is block(1,0), A11 is block(1,1)
        
        # Collect U contributions
        U_rows = []
        U_coeffs = []
        for bi in range(2):
            for bj in range(2):
                coeff = sA[s][bi*2+bj]
                if coeff != 0:
                    idxs = get_A_block_indices(bi, bj)
                    for idx in idxs:
                        U_rows.append(idx)
                        U_coeffs.append(coeff)
                        
        # Determine which B blocks are involved
        # sB[s] gives coeffs for B00, B01, B10, B11
        V_rows = []
        V_coeffs = []
        for bi in range(2):
            for bj in range(2):
                coeff = sB[s][bi*2+bj]
                if coeff != 0:
                    # B_left only uses bj 0,1
                    idxs = get_B_block_indices(bi, bj)
                    for idx in idxs:
                        V_rows.append(idx)
                        V_coeffs.append(coeff)
                        
        # Determine W contributions
        # sC[s] gives coeffs for C00, C01, C10, C11
        # C00 is block(0,0), C01 is block(0,1), C10 is block(1,0), C11 is block(1,1)
        W_rows = []
        W_coeffs = []
        for bi in range(2):
            for bj in range(2):
                coeff = sC[s][bi*2+bj]
                if coeff != 0:
                    idxs = get_C_block_indices(bi, bj)
                    for idx in idxs:
                        W_rows.append(idx)
                        W_coeffs.append(coeff)
                        
        add_term(U_rows, V_rows, W_rows, U_coeffs, V_coeffs, W_coeffs)

    # 2. Schoolbook part for B_right (B02, B12)
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    
    # Term 8: A00 * B02 -> C02
    U_rows = get_A_block_indices(0, 0)
    V_rows = get_B_block_indices(0, 2)
    W_rows = get_C_block_indices(0, 2)
    add_term(U_rows, V_rows, W_rows, [1,1,1,1], [1,1,1,1], [1,1,1,1])
    
    # Term 9: A01 * B12 -> C02
    U_rows = get_A_block_indices(0, 1)
    V_rows = get_B_block_indices(1, 2)
    W_rows = get_C_block_indices(0, 2)
    add_term(U_rows, V_rows, W_rows, [1,1,1,1], [1,1,1,1], [1,1,1,1])
    
    # Term 10: A10 * B02 -> C12
    U_rows = get_A_block_indices(1, 0)
    V_rows = get_B_block_indices(0, 2)
    W_rows = get_C_block_indices(1, 2)
    add_term(U_rows, V_rows, W_rows, [1,1,1,1], [1,1,1,1], [1,1,1,1])
    
    # Term 11: A11 * B12 -> C12
    U_rows = get_A_block_indices(1, 1)
    V_rows = get_B_block_indices(1, 2)
    W_rows = get_C_block_indices(1, 2)
    add_term(U_rows, V_rows, W_rows, [1,1,1,1], [1,1,1,1], [1,1,1,1])

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
