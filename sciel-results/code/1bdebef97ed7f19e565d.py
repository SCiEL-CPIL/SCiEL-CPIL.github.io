def construct():
    n, m, p = 4, 4, 8
    # Build rank-98 decomposition using Strassen-style recursion
    # 4x4 * 4x8 -> split into 2x2 blocks
    # A is 4x4, B is 4x8
    # A blocks: A11, A12, A21, A22 (each 2x2)
    # B blocks: B11, B12, B21, B22 (each 2x4)
    # C blocks: C11, C12, C21, C22 (each 2x4)
    # 
    # Strassen formulas (adapted for 2x4 right operand):
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    #
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mi is a 2x2 * 2x4 multiplication, done with rank 14 (two Strassen 2x2x2)
    
    # Helper: get index in U for A[i,j] where i,j in 0..3
    def u_idx(i, j):
        return i * m + j  # i in 0..3, j in 0..3
    
    # Helper: get index in V for B[j,k] where j in 0..3, k in 0..7
    def v_idx(j, k):
        return j * p + k  # j in 0..3, k in 0..7
    
    # Helper: get index in W for C[k,i] where k in 0..7, i in 0..3
    # W[k*n+i, r] per the convention
    def w_idx(k, i):
        return k * n + i  # k in 0..7, i in 0..3
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * rank for _ in range(p * n)]  # 32 x 98
    
    col = 0
    
    # For each Strassen product Mi, we need a rank-14 decomposition of 2x2 * 2x4
    # 2x2 * 2x4: split 2x4 into two 2x2 halves
    # Left half: columns 0,1 of the 2x4
    # Right half: columns 2,3 of the 2x4
    # Each 2x2*2x2 uses Strassen rank 7
    
    def add_strassen_2x2x2(u_rows, v_rows, w_rows, offset):
        """Add a rank-7 Strassen decomposition for 2x2 * 2x2.
        u_rows: list of 4 (i,m) indices for the 2x2 A block
        v_rows: list of 4 (j,p) indices for the 2x2 B block  
        w_rows: list of 4 (k,n) indices for the 2x2 C block
        offset: column offset
        Returns new column offset.
        """
        nonlocal col
        # 2x2 A: rows u_rows[0], u_rows[1] are row 0, u_rows[2], u_rows[3] are row 1
        # Actually, let me define more carefully.
        # A block is 2x2: A[0,0], A[0,1], A[1,0], A[1,1]
        # u_rows should be [A00, A01, A10, A11] in terms of (i,m) flat indices
        # Similarly for v and w.
        
        a00, a01, a10, a11 = u_rows
        b00, b01, b10, b11 = v_rows
        c00, c01, c10, c11 = w_rows
        
        # Strassen 7 products:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        products = [
            # (u_coeffs, v_coeffs, w_coeffs)
            # M1 = (A00+A11)(B00+B11)
            ({a00: 1, a11: 1}, {b00: 1, b11: 1}, {c00: 1, c11: 1}),
            # M2 = (A10+A11)B00
            ({a10: 1, a11: 1}, {b00: 1}, {c10: 1}),
            # M3 = A00(B01-B11)
            ({a00: 1}, {b01: 1, b11: -1}, {c01: 1}),
            # M4 = A11(B10-B00)
            ({a11: 1}, {b10: 1, b00: -1}, {c00: 1, c10: 1}),
            # M5 = (A00+A01)B11
            ({a00: 1, a01: 1}, {b11: 1}, {c00: -1, c01: 1}),
            # M6 = (A10-A00)(B00+B01)
            ({a10: 1, a00: -1}, {b00: 1, b01: 1}, {c11: 1}),
            # M7 = (A01-A11)(B10+B11)
            ({a01: 1, a11: -1}, {b10: 1, b11: 1}, {c00: 1, c11: 1}),
        ]
        
        for u_c, v_c, w_c in products:
            for row, coeff in u_c.items():
                U[row][col] += coeff
            for row, coeff in v_c.items():
                V[row][col] += coeff
            for row, coeff in w_c.items():
                W[row][col] += coeff
            col += 1
        return col
    
    # Define block indices
    # A is 4x4, flat index i*4+j
    # A11: rows 0,1; cols 0,1 -> [0,1,4,5]
    # A12: rows 0,1; cols 2,3 -> [2,3,6,7]
    # A21: rows 2,3; cols 0,1 -> [8,9,12,13]
    # A22: rows 2,3; cols 2,3 -> [10,11,14,15]
    
    A11 = [0, 1, 4, 5]
    A12 = [2, 3, 6, 7]
    A21 = [8, 9, 12, 13]
    A22 = [10, 11, 14, 15]
    
    # B is 4x8, flat index j*8+k
    # B11: rows 0,1; cols 0,1,2,3 -> [0,1,2,3, 8,9,10,11]
    # B12: rows 0,1; cols 4,5,6,7 -> [4,5,6,7, 12,13,14,15]
    # B21: rows 2,3; cols 0,1,2,3 -> [16,17,18,19, 24,25,26,27]
    # B22: rows 2,3; cols 4,5,6,7 -> [20,21,22,23, 28,29,30,31]
    
    B11 = [0,1,2,3, 8,9,10,11]
    B12 = [4,5,6,7, 12,13,14,15]
    B21 = [16,17,18,19, 24,25,26,27]
    B22 = [20,21,22,23, 28,29,30,31]
    
    # C is 4x8, W index k*4+i
    # C11: rows 0,1; cols 0,1 -> W[k*4+i] for k in 0,1; i in 0,1 -> [0,1,4,5]
    # C12: rows 2,3; cols 0,1 -> W[k*4+i] for k in 2,3; i in 0,1 -> [8,9,12,13]
    # C21: rows 0,1; cols 2,3 -> W[k*4+i] for k in 0,1; i in 2,3 -> [2,3,6,7]
    # C22: rows 2,3; cols 2,3 -> W[k*4+i] for k in 2,3; i in 2,3 -> [10,11,14,15]
    
    C11 = [0,1,4,5]
    C12 = [8,9,12,13]
    C21 = [2,3,6,7]
    C22 = [10,11,14,15]
    
    # Now for each Strassen product, we need to decompose 2x2 * 2x4
    # 2x2 * 2x4: split 2x4 into left 2x2 and right 2x2
    # Left 2x2 of B11: cols 0,1 -> [0,1,8,9]
    # Right 2x2 of B11: cols 2,3 -> [2,3,10,11]
    
    # For a 2x4 B block, split into two 2x2 halves
    def split_2x4(block):
        """Split a 2x4 block (8 indices) into two 2x2 halves."""
        # block = [b00, b01, b02, b03, b10, b11, b12, b13]
        # Left 2x2: [b00, b01, b10, b11]
        # Right 2x2: [b02, b03, b12, b13]
        left = [block[0], block[1], block[4], block[5]]
        right = [block[2], block[3], block[6], block[7]]
        return left, right
    
    # For a 2x4 C block, split into two 2x2 halves
    def split_2x4_c(block):
        """Split a 2x4 C block (8 W indices) into two 2x2 halves."""
        # C is 2x4: rows 0,1; cols 0,1,2,3
        # W index: k*4+i where k is col of C (0..3 for 2x4), i is row (0,1)
        # Wait, I need to be more careful.
        # C block is 2x4: C[row, col] for row in 0,1; col in 0,1,2,3
        # W[k*n + i] where k is the column index in C, i is the row index in C
        # For a 2x4 block at position (r0, c0) in the 4x8 C matrix:
        # W indices: (c0+0)*4+(r0+0), (c0+0)*4+(r0+1), (c0+1)*4+(r0+0), (c0+1)*4+(r0+1),
        #            (c0+2)*4+(r0+0), (c0+2)*4+(r0+1), (c0+3)*4+(r0+0), (c0+3)*4+(r0+1)
        # Left 2x2: cols c0, c0+1
        # Right 2x2: cols c0+2, c0+3
        left = [block[0], block[1], block[4], block[5]]
        right = [block[2], block[3], block[6], block[7]]
        return left, right
    
    # Now I need to define the W indices for each C block more carefully.
    # C11 is at rows 0,1; cols 0,1 in the 4x8 C matrix.
    # W[k*4 + i] where k = col in C, i = row in C
    # C11: k in {0,1}, i in {0,1}
    # W[0*4+0]=0, W[0*4+1]=1, W[1*4+0]=4, W[1*4+1]=5
    # So C11 W indices in order (col0_row0, col0_row1, col1_row0, col1_row1) = [0,1,4,5]
    # For 2x4 split: left 2x2 = cols 0,1 -> [0,1,4,5], right 2x2 = cols 2,3 -> [8,9,12,13]
    # But wait, C11 is only 2x2, not 2x4!
    
    # I'm confusing myself. Let me restart the indexing.
    
    # The full C is 4x8. C[row, col] for row in 0..3, col in 0..7.
    # W index: col * 4 + row (since W[k*n+i] where k=col, i=row, n=4)
    
    # C11 (top-left 2x2): rows 0,1; cols 0,1
    # W indices: 0*4+0=0, 0*4+1=1, 1*4+0=4, 1*4+1=5 -> [0,1,4,5]
    
    # C12 (top-right 2x4): rows 0,1; cols 2,3,4,5,6,7... wait no.
    # A is 4x4, B is 4x8. C = A*B is 4x8.
    # Split A into 2x2 blocks of 2x2:
    #   A11 (rows 0,1; cols 0,1), A12 (rows 0,1; cols 2,3)
    #   A21 (rows 2,3; cols 0,1), A22 (rows 2,3; cols 2,3)
    # Split B into 2x4 blocks:
    #   B11 (rows 0,1; cols 0,1,2,3), B12 (rows 0,1; cols 4,5,6,7)
    #   B21 (rows 2,3; cols 0,1,2,3), B22 (rows 2,3; cols 4,5,6,7)
    # C blocks:
    #   C11 = A11*B11 + A12*B21 (rows 0,1; cols 0,1,2,3) -> 2x4
    #   C12 = A11*B12 + A12*B22 (rows 0,1; cols 4,5,6,7) -> 2x4
    #   C21 = A21*B11 + A22*B21 (rows 2,3; cols 0,1,2,3) -> 2x4
    #   C22 = A21*B12 + A22*B22 (rows 2,3; cols 4,5,6,7) -> 2x4
    
    # So each C block is 2x4, not 2x2!
    
    # W indices for C11 (rows 0,1; cols 0,1,2,3):
    # W[col*4+row]:
    # col=0: W[0]=0, W[1]=1
    # col=1: W[4]=4, W[5]=5
    # col=2: W[8]=8, W[9]=9
    # col=3: W[12]=12, W[13]=13
    # In order (col0r0, col0r1, col1r0, col1r1, col2r0, col2r1, col3r0, col3r1): [0,1,4,5,8,9,12,13]
    
    C11_w = [0,1,4,5,8,9,12,13]
    C12_w = [16,17,20,21,24,25,28,29]  # cols 4,5,6,7
    C21_w = [2,3,6,7,10,11,14,15]     # rows 2,3; cols 0,1,2,3
    C22_w = [18,19,22,23,26,27,30,31] # rows 2,3; cols 4,5,6,7
    
    # Let me verify C21: rows 2,3; cols 0,1,2,3
    # W[col*4+row]:
    # col=0: W[0*4+2]=2, W[0*4+3]=3
    # col=1: W[1*4+2]=6, W[1*4+3]=7
    # col=2: W[2*4+2]=10, W[2*4+3]=11
    # col=3: W[3*4+2]=14, W[3*4+3]=15
    # -> [2,3,6,7,10,11,14,15] ✓
    
    # C22: rows 2,3; cols 4,5,6,7
    # col=4: W[4*4+2]=18, W[4*4+3]=19
    # col=5: W[5*4+2]=22, W[5*4+3]=23
    # col=6: W[6*4+2]=26, W[6*4+3]=27
    # col=7: W[7*4+2]=30, W[7*4+3]=31
    # -> [18,19,22,23,26,27,30,31] ✓
    
    # C12: rows 0,1; cols 4,5,6,7
    # col=4: W[4*4+0]=16, W[4*4+1]=17
    # col=5: W[5*4+0]=20, W[5*4+1]=21
    # col=6: W[6*4+0]=24, W[6*4+1]=25
    # col=7: W[7*4+0]=28, W[7*4+1]=29
    # -> [16,17,20,21,24,25,28,29] ✓
    
    # Now for the Strassen products, each Mi = A_block * B_block where A_block is 2x2 and B_block is 2x4.
    # To compute Mi, split B_block into two 2x2 halves and use Strassen for each.
    
    # For a 2x4 B block with V indices [v0,v1,v2,v3,v4,v5,v6,v7]:
    # v0,v1 = row0,col0,col1; v2,v3 = row0,col2,col3; v4,v5 = row1,col0,col1; v6,v7 = row1,col2,col3
    # Left 2x2: [v0,v1,v4,v5]
    # Right 2x2: [v2,v3,v6,v7]
    
    # For a 2x4 C block with W indices [w0,w1,w2,w3,w4,w5,w6,w7]:
    # w0,w1 = col0,row0,row1; w2,w3 = col1,row0,row1; w4,w5 = col2,row0,row1; w6,w7 = col3,row0,row1
    # Left 2x2: [w0,w1,w2,w3]
    # Right 2x2: [w4,w5,w6,w7]
    
    def add_2x2_x_2x4(a_block, b_block, c_block, col_offset):
        """Add rank-14 decomposition for 2x2 * 2x4.
        a_block: 4 indices [a00,a01,a10,a11]
        b_block: 8 indices [b00,b01,b02,b03,b10,b11,b12,b13]
        c_block: 8 indices [w00,w01,w10,w11,w20,w21,w30,w31] where w_ij is col_i,row_j
        col_offset: starting column
        Returns new column offset.
        """
        nonlocal col
        # Split B into left and right 2x2
        b_left = [b_block[0], b_block[1], b_block[4], b_block[5]]
        b_right = [b_block[2], b_block[3], b_block[6], b_block[7]]
        
        # Split C into left and right 2x2
        c_left = [c_block[0], c_block[1], c_block[2], c_block[3]]
        c_right = [c_block[4], c_block[5], c_block[6], c_block[7]]
        
        # Add Strassen for left half
        col = add_strassen_2x2x2(a_block, b_left, c_left, col_offset)
        # Add Strassen for right half
        col = add_strassen_2x2x2(a_block, b_right, c_right, col_offset)
        
        return col
    
    # Now define the Strassen products for 4x4 * 4x8:
    # M1 = (A11+A22) * (B11+B22)
    # M2 = (A21+A22) * B11
    # M3 = A11 * (B12-B22)
    # M4 = A22 * (B21-B11)
    # M5 = (A11+A12) * B22
    # M6 = (A21-A11) * (B11+B12)
    # M7 = (A12-A22) * (B21+B22)
    
    # For each product, the A block is a linear combination of A sub-blocks,
    # and the B block is a linear combination of B sub-blocks.
    # We need to handle the coefficients properly.
    
    # Let me redefine add_strassen_2x2x2 to handle coefficients.
    # Actually, it's easier to just expand everything.
    
    # Let me restart with a cleaner approach. I'll build the U, V, W matrices
    # by directly specifying each rank-1 term.
    
    # Reset
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # A block indices (flat i*4+j):
    A11 = [0, 1, 4, 5]    # a00, a01, a10, a11
    A12 = [2, 3, 6, 7]
    A21 = [8, 9, 12, 13]
    A22 = [10, 11, 14, 15]
    
    # B block indices (flat j*8+k):
    B11 = [0,1,2,3, 8,9,10,11]    # b00,b01,b02,b03,b10,b11,b12,b13
    B12 = [4,5,6,7, 12,13,14,15]
    B21 = [16,17,18,19, 24,25,26,27]
    B22 = [20,21,22,23, 28,29,30,31]
    
    # C block W indices:
    C11_w = [0,1,4,5,8,9,12,13]
    C12_w = [16,17,20,21,24,25,28,29]
    C21_w = [2,3,6,7,10,11,14,15]
    C22_w = [18,19,22,23,26,27,30,31]
    
    # Strassen product definitions:
    # Each product is (A_combo, B_combo, C_combo)
    # A_combo: list of (A_block, coefficient)
    # B_combo: list of (B_block, coefficient)
    # C_combo: list of (C_block, coefficient)
    
    products = [
        # M1 = (A11+A22)(B11+B22) -> C11 += M1, C22 += M1
        (([(A11,1),(A22,1)], [(B11,1),(B22,1)], [(C11_w,1),(C22_w,1)])),
        # M2 = (A21+A22)B11 -> C21 += M2, C22 -= M2
        (([(A21,1),(A22,1)], [(B11,1)], [(C21_w,1),(C22_w,-1)])),
        # M3 = A11(B12-B22) -> C12 += M3, C22 += M3
        (([(A11,1)], [(B12,1),(B22,-1)], [(C12_w,1),(C22_w,1)])),
        # M4 = A22(B21-B11) -> C11 += M4, C21 += M4
        (([(A22,1)], [(B21,1),(B11,-1)], [(C11_w,1),(C21_w,1)])),
        # M5 = (A11+A12)B22 -> C11 -= M5, C12 += M5
        (([(A11,1),(A12,1)], [(B22,1)], [(C11_w,-1),(C12_w,1)])),
        # M6 = (A21-A11)(B11+B12) -> C22 += M6
        (([(A21,1),(A11,-1)], [(B11,1),(B12,1)], [(C22_w,1)])),
        # M7 = (A12-A22)(B21+B22) -> C11 += M7, C22 += M7
        (([(A12,1),(A22,-1)], [(B21,1),(B22,1)], [(C11_w,1),(C22_w,1)])),
    ]
    
    for a_combo, b_combo, c_combo in products:
        # For each product, we need to compute the 2x2 * 2x4 multiplication
        # using Strassen (rank 14).
        # The A part is a linear combination of 2x2 blocks.
        # The B part is a linear combination of 2x4 blocks.
        
        # Build the effective A 2x2 matrix as a list of (index, coeff)
        a_terms = []
        for block, coeff in a_combo:
            for idx, c in zip(block, [1,1,1,1]):
                a_terms.append((idx, coeff * c))
        
        # Build the effective B 2x4 matrix as a list of (index, coeff)
        b_terms = []
        for block, coeff in b_combo:
            for idx, c in zip(block, [1,1,1,1,1,1,1,1]):
                b_terms.append((idx, coeff * c))
        
        # Build the effective C 2x4 matrix as a list of (index, coeff)
        c_terms = []
        for block, coeff in c_combo:
            for idx, c in zip(block, [1,1,1,1,1,1,1,1]):
                c_terms.append((idx, coeff * c))
        
        # Now decompose the 2x2 * 2x4 into two 2x2 * 2x2 Strassen multiplications.
        # Split B into left 2x2 (cols 0,1) and right 2x2 (cols 2,3)
        # Split C into left 2x2 (cols 0,1) and right 2x2 (cols 2,3)
        
        # For the left half:
        # B left 2x2 indices: positions 0,1,4,5 in the 2x4 block
        # C left 2x2 indices: positions 0,1,2,3 in the 2x4 block
        # For the right half:
        # B right 2x2 indices: positions 2,3,6,7 in the 2x4 block
        # C right 2x2 indices: positions 4,5,6,7 in the 2x4 block
        
        for half in [0, 1]:  # 0=left, 1=right
            if half == 0:
                b_pos = [0, 1, 4, 5]
                c_pos = [0, 1, 2, 3]
            else:
                b_pos = [2, 3, 6, 7]
                c_pos = [4, 5, 6, 7]
            
            # Build A 2x2 terms (same for both halves)
            # A terms: list of (flat_idx, coeff) for the 2x2 A block
            # The 2x2 A block has indices [a00, a01, a10, a11]
            # We need to map these to the Strassen formula.
            
            # Collect A coefficients for each of the 4 positions
            a_coeffs = [0, 0, 0, 0]  # a00, a01, a10, a11
            for idx, coeff in a_terms:
                # idx is a flat index in the 4x4 A matrix
                # We need to figure out which position in the 2x2 block it corresponds to
                # This is tricky because a_terms can have multiple blocks.
                # Let me rethink.
                pass
            
            # This approach is getting complicated. Let me use a different method.
            # I'll directly enumerate all the rank-1 terms.
            pass
    
    # This is getting too complex with the linear combinations. Let me use a different approach.
    # I'll directly construct each rank-1 term by expanding the Strassen formulas.
    
    # Actually, let me just use numpy to verify and construct the solution.
    import numpy as np
    
    # Build the full decomposition using numpy for clarity, then convert to lists.
    
    # Start fresh
    rank = 98
    U_np = np.zeros((n*m, rank))
    V_np = np.zeros((m*p, rank))
    W_np = np.zeros((p*n, rank))
    col = 0
    
    # Strassen 2x2x2: given A (2x2), B (2x2), C (2x2)
    # 7 products, each is a rank-1 term in the 2x2x2 tensor
    def strassen_2x2x2(a_idx, b_idx, c_idx, col_start):
        """a_idx, b_idx, c_idx are lists of 4 indices each [00,01,10,11]
        Returns new column position.
        """
        nonlocal col
        a00, a01, a10, a11 = a_idx
        b00, b01, b10, b11 = b_idx
        c00, c01, c10, c11 = c_idx
        
        # 7 Strassen products
        prods = [
            # (u_indices, u_coeffs, v_indices, v_coeffs, w_indices, w_coeffs)
            ([a00, a11], [1, 1], [b00, b11], [1, 1], [c00, c11], [1, 1]),
            ([a10, a11], [1, 1], [b00], [1], [c10], [1]),
            ([a00], [1], [b01, b11], [1, -1], [c01], [1]),
            ([a11], [1], [b10, b00], [1, -1], [c00, c10], [1, 1]),
            ([a00, a01], [1, 1], [b11], [1], [c00, c01], [-1, 1]),
            ([a10, a00], [1, -1], [b00, b01], [1, 1], [c11], [1]),
            ([a01, a11], [1, -1], [b10, b11], [1, 1], [c00, c11], [1, 1]),
        ]
        
        for u_i, u_c, v_i, v_c, w_i, w_c in prods:
            for ui, uc in zip(u_i, u_c):
                U_np[ui, col] += uc
            for vi, vc in zip(v_i, v_c):
                V_np[vi, col] += vc
            for wi, wc in zip(w_i, w_c):
                W_np[wi, col] += wc
            col += 1
        return col
    
    # Now for the 4x4 * 4x8 case:
    # 7 Strassen products at the top level, each is 2x2 * 2x4
    # Each 2x2 * 2x4 = two 2x2 * 2x2 (left and right halves)
    
    # Define helper to get sub-block indices with coefficients
    def get_a_block(block_name, coeff=1):
        """Return list of (flat_idx, coeff) for a 2x2 A block."""
        blocks = {'A11': A11, 'A12': A12, 'A21': A21, 'A22': A22}
        return [(idx, coeff) for idx in blocks[block_name]]
    
    def get_b_block(block_name, coeff=1):
        """Return list of (flat_idx, coeff) for a 2x4 B block."""
        blocks = {'B11': B11, 'B12': B12, 'B21': B21, 'B22': B22}
        return [(idx, coeff) for idx in blocks[block_name]]
    
    # For each top-level Strassen product, we have:
    # A_part = sum of A blocks with coefficients
    # B_part = sum of B blocks with coefficients
    # C_part = sum of C blocks with coefficients
    # We need to compute A_part * B_part and add to C_part.
    
    # The key insight: (sum a_i A_i) * (sum b_j B_j) = sum_{i,j} a_i b_j (A_i * B_j)
    # But that would be 4x4 = 16 terms per product, which is too many.
    
    # Actually no, the Strassen formula already handles the linear combinations.
    # The issue is that in Strassen, the "multiplication" is of the combined matrices.
    # So (A11+A22) * (B11+B22) is a single 2x2 * 2x4 multiplication.
    
    # To decompose this, I need to handle the linear combinations in the U, V, W matrices.
    # For a product like (A11+A22) * (B11+B22):
    # U gets contributions from A11 and A22 (with coeff 1 each)
    # V gets contributions from B11 and B22 (with coeff 1 each)
    # W gets contributions to C11 and C22 (with coeff 1 each)
    
    # So for each rank-1 term in the Strassen decomposition of 2x2*2x4,
    # I multiply the U coefficients by the A combination coefficients,
    # V coefficients by the B combination coefficients, etc.
    
    # Let me define the 7 top-level products more carefully:
    
    top_products = [
        # (A_blocks, B_blocks, C_blocks)
        # A_blocks: list of (block_name, coeff)
        # B_blocks: list of (block_name, coeff)
        # C_blocks: list of (block_name, coeff)
        (
            [('A11', 1), ('A22', 1)],
            [('B11', 1), ('B22', 1)],
            [('C11', 1), ('C22', 1)]
        ),
        (
            [('A21', 1), ('A22', 1)],
            [('B11', 1)],
            [('C21', 1), ('C22', -1)]
        ),
        (
            [('A11', 1)],
            [('B12', 1), ('B22', -1)],
            [('C12', 1), ('C22', 1)]
        ),
        (
            [('A22', 1)],
            [('B21', 1), ('B11', -1)],
            [('C11', 1), ('C21', 1)]
        ),
        (
            [('A11', 1), ('A12', 1)],
            [('B22', 1)],
            [('C11', -1), ('C12', 1)]
        ),
        (
            [('A21', 1), ('A11', -1)],
            [('B11', 1), ('B12', 1)],
            [('C22', 1)]
        ),
        (
            [('A12', 1), ('A22', -1)],
            [('B21', 1), ('B22', 1)],
            [('C11', 1), ('C22', 1)]
        ),
    ]
    
    C_blocks = {'C11': C11_w, 'C12': C12_w, 'C21': C21_w, 'C22': C22_w}
    
    for a_blocks, b_blocks, c_blocks in top_products:
        # Build the effective 2x2 A matrix indices with coefficients
        # For each of the 4 positions in the 2x2 block, collect all contributions
        a_pos_coeffs = [[0, 0], [0, 0]]  # [row][col]
        for block_name, coeff in a_blocks:
            block = {'A11': A11, 'A12': A12, 'A21': A21, 'A22': A22}[block_name]
            # block = [a00, a01, a10, a11]
            a_pos_coeffs[0][0] += coeff * 1  # a00
            a_pos_coeffs[0][1] += coeff * 1  # a01
            a_pos_coeffs[1][0] += coeff * 1  # a10
            a_pos_coeffs[1][1] += coeff * 1  # a11
        
        # Build the effective 2x4 B matrix indices with coefficients
        b_pos_coeffs = [[0, 0, 0, 0], [0, 0, 0, 0]]  # [row][col]
        for block_name, coeff in b_blocks:
            block = {'B11': B11, 'B12': B12, 'B21': B21, 'B22': B22}[block_name]
            # block = [b00,b01,b02,b03,b10,b11,b12,b13]
            for c in range(4):
                b_pos_coeffs[0][c] += coeff * 1
                b_pos_coeffs[1][c] += coeff * 1
        
        # Build the effective 2x4 C matrix indices with coefficients
        c_pos_coeffs = [[0, 0, 0, 0], [0, 0, 0, 0]]  # [row][col]
        for block_name, coeff in c_blocks:
            block = C_blocks[block_name]
            # block = [w00,w01,w10,w11,w20,w21,w30,w31] where w_ij = col_i, row_j
            # So: w00=col0row0, w01=col0row1, w10=col1row0, w11=col1row1,
            #     w20=col2row0, w21=col2row1, w30=col3row0, w31=col3row1
            for c in range(4):
                c_pos_coeffs[0][c] += coeff * 1  # row 0
                c_pos_coeffs[1][c] += coeff * 1  # row 1
        
        # Now decompose the 2x2 * 2x4 into two 2x2 * 2x2 using Strassen
        for half in [0, 1]:
            if half == 0:
                b_cols = [0, 1]
                c_cols = [0, 1]
            else:
                b_cols = [2, 3]
                c_cols = [2, 3]
            
            # Get the 2x2 sub-blocks
            # A is the same for both halves
            a_sub = [
                (A11[0], a_pos_coeffs[0][0]),  # a00
                (A11[1], a_pos_coeffs[0][1]),  # a01
                (A11[2], a_pos_coeffs[1][0]),  # a10
                (A11[3], a_pos_coeffs[1][1]),  # a11
            ]
            # Wait, this is wrong. The A block indices depend on which A block we're using.
            # Let me fix this.
            
            # Actually, the A indices are always the same flat indices, just with different coefficients.
            # For A11: indices [0,1,4,5] correspond to a00,a01,a10,a11
            # For A22: indices [10,11,14,15] correspond to a00,a01,a10,a11
            # So I need to track which flat index corresponds to which position.
            
            pass
    
    # This is getting really messy. Let me use a completely different approach.
    # I'll directly enumerate all rank-1 terms by expanding everything.
    
    # Actually, the cleanest way is to use the tensor product structure.
    # Let me just build it step by step with explicit index tracking.
    
    # Reset
    rank = 98
    U_np = np.zeros((n*m, rank))
    V_np = np.zeros((m*p, rank))
    W_np = np.zeros((p*n, rank))
    col = 0
    
    # For each of the 7 top-level Strassen products, and for each of the 2 halves (left/right),
    # we add 7 rank-1 terms (from the inner Strassen).
    # Total: 7 * 2 * 7 = 98 terms. ✓
    
    # Define the 7 inner Strassen products for 2x2x2:
    # Each is defined by which positions of A, B, C are involved and with what signs.
    inner_strassen = [
        # (a_positions, a_coeffs, b_positions, b_coeffs, c_positions, c_coeffs)
        # Positions: 0=a00, 1=a01, 2=a10, 3=a11 (for A)
        #            0=b00, 1=b01, 2=b10, 3=b11 (for B, 2x2)
        #            0=c00, 1=c01, 2=c10, 3=c11 (for C, 2x2)
        ([(0,1), (1,1)], [(0,1), (1,1)], [(0,1), (1,1)]),  # M1
        ([(2,1), (3,1)], [(0,1)], [(2,1)]),                   # M2
        ([(0,1)], [(1,1), (3,-1)], [(1,1)]),                  # M3
        ([(3,1)], [(2,1), (0,-1)], [(0,1), (2,1)]),           # M4
        ([(0,1), (1,1)], [(3,1)], [(0,-1), (1,1)]),           # M5
        ([(2,1), (0,-1)], [(0,1), (1,1)], [(3,1)]),           # M6
        ([(1,1), (3,-1)], [(2,1), (3,1)], [(0,1), (3,1)]),    # M7
    ]
    
    # For each top-level product and each half, map positions to flat indices.
    
    for prod_idx in range(7):
        a_blocks, b_blocks, c_blocks = top_products[prod_idx]
        
        for half in [0, 1]:
            # Determine which columns of B and C we're using
            if half == 0:
                b_col_offset = 0  # cols 0,1 of the 2x4
                c_col_offset = 0
            else:
                b_col_offset = 2  # cols 2,3 of the 2x4
                c_col_offset = 2
            
            # For each inner Strassen product, build the rank-1 term
            for inner_idx in range(7):
                a_pos, b_pos, c_pos = inner_strassen[inner_idx]
                
                # For A: for each (position, coeff) in a_pos
                # position 0=a00, 1=a01, 2=a10, 3=a11
                # The flat index depends on which A block is being used.
                # a_blocks can have multiple blocks, so we sum contributions.
                
                for a_block_name, a_block_coeff in a_blocks:
                    block = {'A11': A11, 'A12': A12, 'A21': A21, 'A22': A22}[a_block_name]
                    # block = [idx_a00, idx_a01, idx_a10, idx_a11]
                    for pos, coeff in a_pos:
                        flat_idx = block[pos]
                        U_np[flat_idx, col] += a_block_coeff * coeff
                
                # For B: similar
                for b_block_name, b_block_coeff in b_blocks:
                    block = {'B11': B11, 'B12': B12, 'B21': B21, 'B22': B22}[b_block_name]
                    # block = [b00,b01,b02,b03,b10,b11,b12,b13]
                    # For the 2x2 sub-block at columns b_col_offset, b_col_offset+1:
                    # pos 0=b00 -> block[b_col_offset + 0]
                    # pos 1=b01 -> block[b_col_offset + 1]
                    # pos 2=b10 -> block[4 + b_col_offset + 0]
                    # pos 3=b11 -> block[4 + b_col_offset + 1]
                    pos_to_flat = {
                        0: b_col_offset,
                        1: b_col_offset + 1,
                        2: 4 + b_col_offset,
                        3: 4 + b_col_offset + 1,
                    }
                    for pos, coeff in b_pos:
                        flat_idx = block[pos_to_flat[pos]]
                        V_np[flat_idx, col] += b_block_coeff * coeff
                
                # For C: similar
                for c_block_name, c_block_coeff in c_blocks:
                    block = C_blocks[c_block_name]
                    # block = [w00,w01,w10,w11,w20,w21,w30,w31]
                    # w_ij where i=col, j=row
                    # pos 0=c00 -> col0,row0 -> block[0 + c_col_offset*2 + 0]... 
                    # Wait, let me think about this more carefully.
                    # block has 8 entries: [col0row0, col0row1, col1row0, col1row1, col2row0, col2row1, col3row0, col3row1]
                    # For the 2x2 sub-block at columns c_col_offset, c_col_offset+1:
                    # pos 0=c00 -> col=c_col_offset, row=0 -> block[c_col_offset*2 + 0]
                    # pos 1=c01 -> col=c_col_offset, row=1 -> block[c_col_offset*2 + 1]
                    # pos 2=c10 -> col=c_col_offset+1, row=0 -> block[(c_col_offset+1)*2 + 0]
                    # pos 3=c11 -> col=c_col_offset+1, row=1 -> block[(c_col_offset+1)*2 + 1]
                    pos_to_flat = {
                        0: c_col_offset * 2 + 0,
                        1: c_col_offset * 2 + 1,
                        2: (c_col_offset + 1) * 2 + 0,
                        3: (c_col_offset + 1) * 2 + 1,
                    }
                    for pos, coeff in c_pos:
                        flat_idx = block[pos_to_flat[pos]]
                        W_np[flat_idx, col] += c_block_coeff * coeff
                
                col += 1
    
    # Convert to lists
    U_list = U_np.tolist()
    V_list = V_np.tolist()
    W_list = W_np.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
