"""Matrix multiplication 6x6x6 - structural decomposition using Strassen blocks."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 6x6 matrix multiplication using 2x2 blocks of 3x3 matrices
    with Strassen's algorithm for the block structure.
    
    Rank = 7 * 27 = 189 (using schoolbook for 3x3 blocks)
    vs schoolbook 216.
    """
    import time
    start = time.time()
    
    n = m = p = 6
    # Block sizes
    b = 3  # block dimension
    
    # Strassen's algorithm for 2x2 block matrices:
    # We have A = [[A11, A12], [A21, A22]] and B = [[B11, B12], [B21, B22]]
    # Product C = [[C11, C12], [C21, C22]]
    #
    # Strassen computes 7 products:
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    #
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mi is a 3x3 matrix product.
    # For 3x3 schoolbook: rank 27
    # Total rank: 7 * 27 = 189
    
    block_rank = 27  # schoolbook rank for 3x3
    
    total_rank = 7 * block_rank
    
    # Build U, V, W matrices
    # U has shape (n*m, total_rank) = (36, 189)
    # V has shape (m*p, total_rank) = (36, 189)  
    # W has shape (p*n, total_rank) = (36, 189)
    
    # Index conventions:
    # U[i*m + j, col] - row index for A[i,j]
    # V[j*p + k, col] - row index for B[j,k]
    # W[k*n + i, col] - row index for C[i,k] (output factor follows W[k*n+i, r])
    
    # For each Strassen multiplication Mi, we need to express it as
    # a 3x3 matrix product with specific linear combinations of blocks.
    #
    # Let's define the 4 quadrants for A (input matrix 1):
    # A11: rows 0-2, cols 0-2
    # A12: rows 0-2, cols 3-5
    # A21: rows 3-5, cols 0-2
    # A22: rows 3-5, cols 3-5
    #
    # For B (input matrix 2):
    # B11: rows 0-2, cols 0-2
    # B12: rows 0-2, cols 3-5
    # B21: rows 3-5, cols 0-2
    # B22: rows 3-5, cols 3-5
    #
    # For C (output):
    # C11: rows 0-2, cols 0-2
    # C12: rows 0-2, cols 3-5
    # C21: rows 3-5, cols 0-2
    # C22: rows 3-5, cols 3-5
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # Helper: add a 3x3 schoolbook decomposition starting at column offset
    def add_schoolbook_3x3(U, V, W, col_offset, 
                           # A block offsets (row, col) in the 6x6 matrix
                           a_row_offsets, a_col_offsets,
                           # B block offsets (row, col) in the 6x6 matrix  
                           b_row_offsets, b_col_offsets,
                           # C block offsets (row, col) in the 6x6 matrix
                           c_row_offsets, c_col_offsets,
                           # Signs for A block
                           a_signs,
                           # Signs for B block
                           b_signs):
        """
        Add a schoolbook 3x3 decomposition where:
        - A part is sum of a_row_offsets[r]*a_signs[r] * A_block(r)
        - B part is sum of b_row_offsets[c]*b_signs[c] * B_block(c)
        - C part collects into c_row_offsets/c_col_offsets
        """
        for r in range(3):
            for c in range(3):
                col = col_offset + r * 3 + c
                
                # A factors: for each element (r,c) in the 3x3 product
                # A element is at position depending on which block
                # We need to map 3x3 indices to 6x6 positions
                
                # For each contributing A block
                for ai, aro in enumerate(a_row_offsets):
                    a_sign = a_signs[ai]
                    if a_sign == 0:
                        continue
                    # A block at rows aro, cols a_col_offsets[ai]
                    # Element (r,c) in this 3x3 block maps to:
                    # row = aro + r, col = a_col_offsets[ai] + c in 6x6
                    U_row = (aro + r) * m + (a_col_offsets[ai] + c)
                    U[U_row][col] += a_sign
                
                # B factors: for each contributing B block
                for bi, bro in enumerate(b_row_offsets):
                    b_sign = b_signs[bi]
                    if b_sign == 0:
                        continue
                    # B block at rows bro, cols b_col_offsets[bi]
                    # Element (r,c) in this 3x3 block maps to:
                    # row = bro + r, col = b_col_offsets[bi] + c in 6x6
                    V_row = (bro + r) * p + (b_col_offsets[bi] + c)
                    V[V_row][col] += b_sign
                
                # C factors: output goes to c block
                for ci, cro in enumerate(c_row_offsets):
                    # C block at rows cro, cols c_col_offsets[ci]
                    # Element (r,c) in this 3x3 block maps to:
                    # row = cro + r, col = c_col_offsets[ci] + c in 6x6
                    W_row = (c_col_offsets[ci] + c) * n + (cro + r)
                    W[W_row][col] += 1
    
    # Strassen M1 = (A11 + A22) * (B11 + B22)
    # A: A11 (rows 0-2, cols 0-2) + A22 (rows 3-5, cols 3-5)
    # B: B11 (rows 0-2, cols 0-2) + B22 (rows 3-5, cols 3-5)
    # C: C11 and C22 contributions handled later
    # For M1, we don't directly assign to C; we track it separately
    # Actually, let's just build the full decomposition directly
    
    # Let me reconsider: each Mi is a full 3x3 matrix product.
    # The schoolbook for 3x3 gives 27 scalar multiplications.
    # Each scalar multiplication corresponds to one rank-1 term.
    
    # For a 3x3 product X*Y=Z where X and Y are 3x3:
    # Z[r][c] = sum_d X[r][d] * Y[d][c]
    # Schoolbook: for each (r,c,d): one rank-1 term
    # U factor: X[r][d] -> A indices
    # V factor: Y[d][c] -> B indices  
    # W factor: Z[r][c] -> C indices
    
    # Let me redefine more carefully.
    
    # For a general 3x3 block product where:
    # X is a linear combination of A blocks
    # Y is a linear combination of B blocks
    # Z contributes to C blocks
    
    # The 27 terms for 3x3 schoolbook are indexed by (r, d, c) in 0..2
    # X[r][d] * Y[d][c] contributes to Z[r][c]
    
    def add_block_product(col_offset, 
                         x_blocks, y_blocks, z_blocks,
                         x_signs, y_signs):
        """
        Add a 3x3 block product to the decomposition.
        
        x_blocks: list of (row_offset, col_offset) for X matrix blocks in A
        y_blocks: list of (row_offset, col_offset) for Y matrix blocks in B
        z_blocks: list of (row_offset, col_offset) for Z matrix blocks in C
        x_signs: signs for each X block
        y_signs: signs for each Y block
        """
        for r in range(3):
            for d in range(3):
                for c in range(3):
                    col = col_offset + r * 9 + d * 3 + c
                    
                    # X[r][d] contributions
                    for xi, (xo_r, xo_c) in enumerate(x_blocks):
                        s = x_signs[xi]
                        if s == 0:
                            continue
                        # X block element (r,d) is at A position (xo_r+r, xo_c+d)
                        U_row = (xo_r + r) * m + (xo_c + d)
                        U[U_row][col] += s
                    
                    # Y[d][c] contributions
                    for yi, (yo_r, yo_c) in enumerate(y_blocks):
                        s = y_signs[yi]
                        if s == 0:
                            continue
                        # Y block element (d,c) is at B position (yo_r+d, yo_c+c)
                        V_row = (yo_r + d) * p + (yo_c + c)
                        V[V_row][col] += s
                    
                    # Z[r][c] contributions
                    for zi, (zo_r, zo_c) in enumerate(z_blocks):
                        # Z block element (r,c) is at C position (zo_r+r, zo_c+c)
                        # W index: k*n + i where k=col, i=row
                        W_row = (zo_c + c) * n + (zo_r + r)
                        W[W_row][col] += 1
    
    # Now define the 7 Strassen products
    
    # Block positions in 6x6:
    # Quadrant 1 (top-left): rows 0-2, cols 0-2
    # Quadrant 2 (top-right): rows 0-2, cols 3-5
    # Quadrant 3 (bottom-left): rows 3-5, cols 0-2
    # Quadrant 4 (bottom-right): rows 3-5, cols 3-5
    
    # M1 = (A11 + A22) * (B11 + B22)
    add_block_product(
        0,
        [(0,0), (3,3)],  # A11, A22
        [(0,0), (3,3)],  # B11, B22
        [],  # Z not directly assigned yet - we'll handle C accumulation separately
        [1, 1],
        [1, 1]
    )
    
    # Wait, this approach doesn't work cleanly because C is shared across multiple M's.
    # Let me think differently.
    
    # Actually, the standard way: each Mi produces a 3x3 matrix.
    # Then C quadrants are linear combinations of the Mi's.
    # 
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # So for each Mi, it contributes to specific C quadrants with specific signs.
    #
    # Let me redo this. For each Mi, I add 27 terms (3x3 schoolbook).
    # Each term has U, V factors from the A and B blocks, and W factor
    # that goes to the appropriate C quadrants with the right sign.
    
    # Clear and restart
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    def add_strassen_product(col_offset,
                            a_blocks, a_signs,
                            b_blocks, b_signs,
                            c_contribs):
        """
        Add one Strassen product (3x3 schoolbook, 27 terms).
        
        a_blocks: list of (row_off, col_off) for A sub-blocks
        a_signs: corresponding signs
        b_blocks: list of (row_off, col_off) for B sub-blocks  
        b_signs: corresponding signs
        c_contribs: list of (c_quad_row, c_quad_col, sign) for how this M contributes to C
        """
        for r in range(3):
            for d in range(3):
                for c in range(3):
                    col = col_offset + r * 9 + d * 3 + c
                    
                    # U: from A blocks
                    for ai, (ar, ac) in enumerate(a_blocks):
                        s = a_signs[ai]
                        if s == 0:
                            continue
                        U_row = (ar + r) * m + (ac + d)
                        U[U_row][col] += s
                    
                    # V: from B blocks
                    for bi, (br, bc) in enumerate(b_blocks):
                        s = b_signs[bi]
                        if s == 0:
                            continue
                        V_row = (br + d) * p + (bc + c)
                        V[V_row][col] += s
                    
                    # W: from C contributions
                    for (cr, cc, cs) in c_contribs:
                        if cs == 0:
                            continue
                        W_row = (cc + c) * n + (cr + r)
                        W[W_row][col] += cs
    
    # M1 = (A11 + A22) * (B11 + B22)
    # Contributes to: C11 (+1), C22 (+1)
    add_strassen_product(
        0,
        [(0,0), (3,3)], [1, 1],
        [(0,0), (3,3)], [1, 1],
        [(0,0, 1), (3,3, 1)]  # C11 +, C22 +
    )
    
    # M2 = (A21 + A22) * B11
    # Contributes to: C21 (+1), C22 (-1)
    add_strassen_product(
        27,
        [(3,0), (3,3)], [1, 1],
        [(0,0)], [1],
        [(3,0, 1), (3,3, -1)]  # C21 +, C22 -
    )
    
    # M3 = A11 * (B12 - B22)
    # Contributes to: C12 (+1), C22 (+1)
    add_strassen_product(
        54,
        [(0,0)], [1],
        [(0,3), (3,3)], [1, -1],
        [(0,3, 1), (3,3, 1)]  # C12 +, C22 +
    )
    
    # M4 = A22 * (B21 - B11)
    # Contributes to: C11 (-1), C21 (+1)
    add_strassen_product(
        81,
        [(3,3)], [1],
        [(3,0), (0,0)], [1, -1],
        [(0,0, -1), (3,0, 1)]  # C11 -, C21 +
    )
    
    # M5 = (A11 + A12) * B22
    # Contributes to: C11 (+1), C12 (+1)
    add_strassen_product(
        108,
        [(0,0), (0,3)], [1, 1],
        [(3,3)], [1],
        [(0,0, 1), (0,3, 1)]  # C11 +, C12 +
    )
    
    # M6 = (A21 - A11) * (B11 + B12)
    # Contributes to: C22 (+1)
    add_strassen_product(
        135,
        [(3,0), (0,0)], [1, -1],
        [(0,0), (0,3)], [1, 1],
        [(3,3, 1)]  # C22 +
    )
    
    # M7 = (A12 - A22) * (B21 + B22)
    # Contributes to: C11 (-1)
    add_strassen_product(
        162,
        [(0,3), (3,3)], [1, -1],
        [(3,0), (3,3)], [1, 1],
        [(0,0, -1)]  # C11 -
    )
    
    # Verify we haven't exceeded time
    if time.time() - start > 140:
        pass  # Return what we have
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
