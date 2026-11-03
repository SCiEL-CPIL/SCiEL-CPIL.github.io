"""Matrix multiplication 2x4x8 - rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen's algorithm on 2x2 blocks
    # 2x4x8 can be viewed as 8 independent 2x2x2 multiplications
    # Each 2x2x2 uses rank 7 (Strassen)
    # Total rank: 8 * 7 = 56
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen's 7 multiplications for 2x2x2
    # Each term: (U_coeffs, V_coeffs, W_coeffs)
    # Indices are (i,j) for 2x2 matrices
    strassen_terms = [
        # M1 = (A00+A11)(B00+B11)
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        # M2 = (A10+A11)B00
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        # M3 = A00(B01-B11)
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        # M4 = A11(B10-B00)
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        # M5 = (A00+A01)B11
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        # M6 = (A10-A00)(B00+B01)
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        # M7 = (A01-A11)(B10+B11)
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1})
    ]
    
    col = 0
    # 8 blocks: 4 for left half (columns 0-3 of B), 4 for right half (columns 4-7)
    for block in range(8):
        is_left = block < 4
        block_idx = block % 4
        
        # A is 2x4: rows 0-1, columns 0-3
        # For left blocks, use columns 0-1; for right blocks, use columns 2-3
        if is_left:
            a_j_offset = 0
        else:
            a_j_offset = 2
        
        # B is 4x8: rows 0-3, columns 0-7
        # For left blocks, use rows 0-1; for right blocks, use rows 2-3
        if is_left:
            b_i_offset = 0
        else:
            b_i_offset = 2
        b_j_offset = block_idx * 2
        
        # C is 2x8: rows 0-1, columns 0-7
        # Columns match B's column offsets
        c_j_offset = block_idx * 2
        
        for u_coeffs, v_coeffs, w_coeffs in strassen_terms:
            # Fill U: A is 2x4, index is i*m + j
            for (li, lj), val in u_coeffs.items():
                gi = li  # row of A (0 or 1)
                gj = a_j_offset + lj  # column of A
                U[gi * m + gj][col] = val
            
            # Fill V: B is 4x8, index is i*p + j
            for (li, lj), val in v_coeffs.items():
                gi = b_i_offset + li  # row of B
                gj = b_j_offset + lj  # column of B
                V[gi * p + gj][col] = val
            
            # Fill W: C is 2x8, but stored as W[k*n+i,r] where k is column, i is row
            for (li, lj), val in w_coeffs.items():
                gi = li  # row of C (0 or 1)
                gj = c_j_offset + lj  # column of C
                # W index: k*n + i = gj*n + gi
                W[gj * n + gi][col] = val
            
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
