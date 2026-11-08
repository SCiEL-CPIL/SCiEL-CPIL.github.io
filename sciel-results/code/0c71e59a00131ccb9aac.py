"""Strassen-based rank reduction for 4x4 * 4x8 matrix multiplication."""
DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # We decompose into 2x2 blocks of size 2x2 (A), 2x4 (B), 2x4 (C)
    # A: 4x4 -> blocks A00, A01, A10, A11 each 2x2
    # B: 4x8 -> blocks B00, B01, B02, B03 each 2x4 (but actually 2x4 means 2 rows, 4 cols)
    # Wait: B is 4x8, so 2x2 blocks would be 2x4? No.
    # B is 4x8: if we split rows into 2 groups of 2, and cols into 2 groups of 4:
    #   B00: rows 0-1, cols 0-3 (2x4)
    #   B01: rows 0-1, cols 4-7 (2x4)
    #   B10: rows 2-3, cols 0-3 (2x4)
    #   B11: rows 2-3, cols 4-7 (2x4)
    # C = A*B: 4x8
    #   C00: rows 0-1, cols 0-3 = A00*B00 + A01*B10 (2x4)
    #   C01: rows 0-1, cols 4-7 = A00*B01 + A01*B11 (2x4)
    #   C10: rows 2-3, cols 0-3 = A10*B00 + A11*B10 (2x4)
    #   C11: rows 2-3, cols 4-7 = A10*B01 + A11*B11 (2x4)
    
    # So we have 4 independent 2x2 * 2x4 multiplications to combine.
    # Each 2x2 * 2x4 can be done with rank 14 (vs 16 schoolbook) using a known decomposition.
    # Total: 4 * 14 = 56? But we can do better with Strassen-style sharing.
    
    # Actually, let me think more carefully. The full 4x4*4x8 tensor has 4*4*8=128 components.
    # 
    # Better approach: Use the fact that 4x4*4x8 can be seen as two independent 4x4*4x4 
    # multiplications (for the two halves of the 8 columns). Each 4x4*4x4 can use Strassen rank 22 (known best for 4x4).
    # So total rank = 2 * 22 = 44.
    
    # Even better: known rank for 4x4 matrix mult is 22 (over reals).
    # For 4x4 * 4x8, we can do it as two 4x4*4x4 with shared A:
    # C = A * B1 || A * B2 where B1, B2 are 4x4
    # This requires rank(r(4x4) + r(4x4)) = 2*22 = 44 if we share nothing,
    # but since A is shared, we can potentially do better.
    # Actually the tensor for 4x4*4x8 is not simply two copies; it's a single tensor.
    # The rank of the 4x4*4x8 multiplication tensor is at most 4*4*8=128 (schoolbook).
    # 
    # Let me use a block Strassen approach:
    # Split A (4x4) into four 2x2 blocks, B (4x8) into four 2x4 blocks.
    # Use Strassen's 7 products for the 2x2 block structure, but each "product" is a 2x2 * 2x4 mult.
    # A 2x2 * 2x4 mult has rank at most 8 (schoolbook: 2*2*4=16, but we can do better).
    # Actually 2x2*2x4: this is like multiplying a 2x2 by a 2x4.
    # The tensor rank of 2x2*2x4: schoolbook is 2*2*4=16.
    # But we can use: C = A*B where A is 2x2, B is 2x4.
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1]
    # etc.
    # This is essentially 2x2 matrix multiplication applied column-wise, so rank = 2 * r(2x2) = 2*7 = 14? No.
    # Actually, the 2x2*2x4 tensor is isomorphic to two independent 2x2*2x2 multiplications (one for each pair of output columns), so rank = 2*7 = 14.
    
    # So with Strassen: 7 Strassen "multiplications", each of rank 14, total rank = 7*14 = 98.
    # That's worse than 44.
    
    # Let me just use the known result: rank(4x4*4x4) = 22 over R.
    # For 4x4*4x8, I'll use a construction that gives rank 44.
    # 
    # Construct: C = A*B where A is 4x4, B is 4x8.
    # Write B = [B1 | B2] where B1, B2 are 4x4.
    # C = [A*B1 | A*B2]
    # Use rank-22 decomposition for each 4x4*4x4.
    # But the A factors are the same, so we can potentially share.
    # However, in tensor decomposition terms, the 4x4*4x8 tensor lives in (R^16* x R^32* x R^32) or similar.
    # The rank is at most 44 by doing two independent 4x4*4x4 decompositions.
    #
    # I need explicit rank-22 decomposition for 4x4 matrix multiplication.
    # This is complex. Let me use a simpler but still improved approach.
    #
    # Alternative: Use the fact that 4 = 2^2 and apply Strassen recursively.
    # 4x4*4x4 via Strassen: treat as 2x2 blocks of 2x2.
    # Strassen gives 7 products of 2x2*2x2, each of rank 7, so total 7*7=49.
    # But known best for 4x4 is 22, so this is far from optimal.
    #
    # For a practical improvement over 128, let me use the recursive Strassen: rank 49 for 4x4*4x4,
    # and for 4x4*4x8, use two such with shared A, giving rank 2*49 = 98? No, that's not right either.
    #
    # Let me just implement a clean recursive Strassen that gives rank 98 for the full problem.
    # Actually, for 4x4 * 4x8:
    # Split A into 2x2 blocks (four 2x2 blocks), B into 2x2 blocks (four 2x4 blocks... no).
    # 
    # Let me be very explicit. I'll implement a tensor decomposition using nested Strassen.
    
    # For now, let me use a simpler approach: schoolbook but with some structure exploited.
    # The minimum known rank for 4x4*4x8 is likely around 44-50.
    # 
    # Let me implement a rank-49 construction using Strassen for the 4x4*4x4 parts and
    # handle the 4x8 by splitting into two 4x4.
    
    # Actually, the simplest strong improvement: use the identity that
    # matmul(4,4,8) can be embedded in matmul(4,4,4) x matmul(4,4,4) with shared input A.
    # The tensor rank of matmul(4,4,8) <= rank(matmul(4,4,4)) + rank(matmul(4,4,4)) = 2*r(4,4,4).
    # With r(4,4,4) = 49 (Strassen), we get rank 98.
    #
    # Let me build this explicitly.
    
    # Strassen for 2x2*2x2: rank 7
    # Recursive: 4x4*4x4 = 7 copies of 2x2*2x2, each rank 7, total 49.
    # For 4x4*4x8: split B into B_left (4x4) and B_right (4x4).
    # C_left = A * B_left, C_right = A * B_right.
    # Use Strassen decomposition for each, sharing the A-side structure.
    
    # I'll build the full 4x4*4x4 Strassen decomposition first, then duplicate for the two halves.
    
    # Strassen 2x2*2x2:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 4x4*4x4, apply Strassen at the 2x2 block level.
    # A has blocks A00,A01,A10,A11 each 2x2.
    # B has blocks B00,B01,B10,B11 each 2x2.
    # Each "product" like (A00+A11)(B00+B11) is a 2x2*2x2 multiplication, which itself needs rank 7.
    
    # So for 4x4*4x4: 7 Strassen products, each of rank 7 (for 2x2*2x2), total rank 49.
    
    # For 4x4*4x8: B is 4x8. Split B into B1 (cols 0-3) and B2 (cols 4-7), each 4x4.
    # C1 = A*B1 (4x4), C2 = A*B2 (4x4).
    # Each needs rank 49, total rank 98.
    
    # But wait, the A factors in both C1 and C2 use the same A, so in the tensor decomposition,
    # the U matrix (for A) will have duplicate rows. That's fine; rank is still 98.
    
    # Let me build this.
    
    def strassen_2x2_2x2():
        """Return (U, V, W) for 2x2*2x2, rank 7.
        U: (4, 7), V: (4, 7), W: (4, 7)
        Indexing: A[i,j] -> U[i*2+j, :], B[i,j] -> V[i*2+j, :], C[i,j] -> W[i*2+j, :]
        """
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1  # C00,M4,-M5+M7; C01,M3+M5; C10,M2+M4; C11,M1-M2+M3+M6
        # Wait, W accumulates. Let me redo.
        # For M1: contributes +1 to C00, +1 to C11
        W[0][0] = 1; W[3][0] = 1
        # M2 = (A10+A11)B00: contributes +1 to C10, -1 to C11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = -1
        # M3 = A00(B01-B11): contributes +1 to C01, +1 to C11
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        # M4 = A11(B10-B00): contributes +1 to C00, +1 to C10
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        # M5 = (A00+A01)B11: contributes -1 to C00, +1 to C01
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        # M6 = (A10-A00)(B00+B01): contributes +1 to C11
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        # M7 = (A01-A11)(B10+B11): contributes +1 to C00
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1
        return U, V, W
    
    U22, V22, W22 = strassen_2x2_2x2()
    
    # Now build 4x4*4x4 using Strassen at block level.
    # A is 4x4, split into four 2x2 blocks: A00,A01,A10,A11
    # B is 4x4, split into four 2x2 blocks: B00,B01,B10,B11
    # Strassen products:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # Each Mi is a 2x2*2x2 product, using U22, V22, W22.
    
    # For 4x4*4x8, B is 4x8. Split into B1 (cols 0-3) and B2 (cols 4-7), each 4x4.
    # C1 = A*B1, C2 = A*B2.
    # Use the same Strassen structure for both.
    
    # Total rank = 7 (Strassen) * 7 (2x2 rank) * 2 (two halves) = 98.
    
    rank = 98
    nU = 16  # 4*4
    nV = 32  # 4*8
    nW = 32  # 4*8
    
    U = [[0]*rank for _ in range(nU)]
    V = [[0]*rank for _ in range(nV)]
    W = [[0]*rank for _ in range(nW)]
    
    # Helper: add a 2x2*2x2 Strassen decomposition at a given offset
    def add_strassen_block(u_base, v_base, w_base, col_offset):
        """Add Strassen 2x2*2x2 contribution starting at column col_offset.
        u_base: (row_offset for A) - index into A's 16 elements
        v_base: (row_offset for B) - index into B's 32 elements  
        w_base: (row_offset for C) - index into C's 32 elements
        """
        for c in range(7):
            for i in range(4):
                if U22[i][c] != 0:
                    U[u_base + i][col_offset + c] += U22[i][c]
                if V22[i][c] != 0:
                    V[v_base + i][col_offset + c] += V22[i][c]
                if W22[i][c] != 0:
                    W[w_base + i][col_offset + c] += W22[i][c]
    
    # Strassen combinations at the 2x2 block level for 4x4*4x4:
    # For each of the 7 Strassen products, we need to define the A-block combination and B-block combination.
    
    # A blocks: A00 (rows 0-1, cols 0-1), A01 (rows 0-1, cols 2-3), A10 (rows 2-3, cols 0-1), A11 (rows 2-3, cols 2-3)
    # In linear indexing (row-major, 4x4): 
    #   A00: indices 0,1,4,5
    #   A01: indices 2,3,6,7
    #   A10: indices 8,9,12,13
    #   A11: indices 10,11,14,15
    # For B (4x4 part): B00: 0,1,4,5; B01: 2,3,6,7; B10: 8,9,12,13; B11: 10,11,14,15
    
    # For the full 4x4*4x8, B has 32 elements. 
    # B1 (cols 0-3): elements 0-15 in V
    # B2 (cols 4-7): elements 16-31 in V
    
    # Strassen products for C1 = A*B1:
    # M1 = (A00+A11)(B1_00+B1_11)
    # M2 = (A10+A11)B1_00
    # M3 = A00(B1_01-B1_11)
    # M4 = A11(B1_10-B1_00)
    # M5 = (A00+A01)B1_11
    # M6 = (A10-A00)(B1_00+B1_01)
    # M7 = (A01-A11)(B1_10+B1_11)
    
    # For each Mi, the A-part is a linear combination of A-blocks, and B-part is a linear combination of B-blocks.
    # Each "product" of combined blocks is itself a 2x2*2x2 multiplication.
    
    # I'll define the A and B block coefficients for each Strassen product.
    
    # A-block indices: 0=A00, 1=A01, 2=A10, 3=A11
    # A-block row offsets in U (each block is 2x2=4 elements):
    # A00: row 0, A01: row 2, A10: row 8, A11: row 10 (in the 16-row U matrix)
    # Wait, U has 16 rows for A (4x4). Block A00 occupies rows 0,1,4,5? No.
    # A is 4x4, linearized as A[i*4+j]. 
    # A00 = A[0:2,0:2] = elements (0,0),(0,1),(1,0),(1,1) = indices 0,1,4,5
    # A01 = A[0:2,2:4] = elements (0,2),(0,3),(1,2),(1,3) = indices 2,3,6,7
    # A10 = A[2:4,0:2] = elements (2,0),(2,1),(3,0),(3,1) = indices 8,9,12,13
    # A11 = A[2:4,2:4] = elements (2,2),(2,3),(3,2),(3,3) = indices 10,11,14,15
    
    # For the U matrix (16 x rank), each row corresponds to one element of A.
    # To form A00+A11, I need to add rows 0,1,4,5 and 10,11,14,15.
    # But in the tensor decomposition, U[u, c] gives the coefficient of A[u] in the c-th product.
    # So for product M1 = (A00+A11)(B1_00+B1_11), the U part has 1's in rows 0,1,4,5,10,11,14,15.
    # But that's not how it works. The Strassen formula says M1 = (A00+A11)*(B00+B11).
    # This means we're multiplying the sum (A00+A11) by the sum (B00+B11).
    # In tensor decomposition, this is a single bilinear form: sum over all elements.
    # So U has 1 in all rows corresponding to A00 and A11, and V has 1 in all rows corresponding to B00 and B11.
    # But then the product is (A00+A11) * (B00+B11) which is a 2x2 matrix, and we need to extract C from it.
    # This is exactly what the inner Strassen (U22,V22,W22) does.
    
    # So the structure is:
    # For each of the 7 outer Strassen products:
    #   Define A_coeff[16] (coefficients for each A element)
    #   Define B_coeff[16] (coefficients for each B element in the 4x4 part)
    #   The product P = A_coeff^T * A * B * B_coeff (conceptually)
    #   But P is a 2x2 matrix (the result of multiplying the combined A-block by combined B-block).
    #   Then C gets contributions from P via the W22 structure.
    
    # Actually, this is getting complex. Let me use a different approach.
    # 
    # I'll directly construct the rank-98 decomposition by iterating over all 98 products.
    
    # For C1 = A*B1 (4x4*4x4):
    # 7 outer Strassen products, each using 7 inner Strassen products = 49 products.
    # For C2 = A*B2 (4x4*4x4): another 49 products.
    # Total: 98 products.
    
    # Outer Strassen A-block coefficients (for 4x4 A, 2x2 blocks):
    # Each outer product combines A-blocks with certain coefficients.
    # A-block to element mapping:
    # A00: [0,1,4,5], A01: [2,3,6,7], A10: [8,9,12,13], A11: [10,11,14,15]
    
    a_block_elems = {
        0: [0, 1, 4, 5],   # A00
        1: [2, 3, 6, 7],   # A01
        2: [8, 9, 12, 13], # A10
        3: [10, 11, 14, 15] # A11
    }
    
    # Outer Strassen products (A-block coefficients, B-block coefficients):
    # M1: A: {0:1, 3:1}, B: {0:1, 3:1}
    # M2: A: {2:1, 3:1}, B: {0:1}
    # M3: A: {0:1}, B: {1:1, 3:-1}
    # M4: A: {3:1}, B: {2:1, 0:-1}
    # M5: A: {0:1, 1:1}, B: {3:1}
    # M6: A: {2:1, 0:-1}, B: {0:1, 1:1}
    # M7: A: {1:1, 3:-1}, B: {2:1, 3:1}
    
    outer_products = [
        ({0:1, 3:1}, {0:1, 3:1}),    # M1
        ({2:1, 3:1}, {0:1}),          # M2
        ({0:1}, {1:1, 3:-1}),         # M3
        ({3:1}, {2:1, 0:-1}),         # M4
        ({0:1, 1:1}, {3:1}),          # M5
        ({2:1, 0:-1}, {0:1, 1:1}),    # M6
        ({1:1, 3:-1}, {2:1, 3:1}),    # M7
    ]
    
    # C-block contributions from each outer product:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    c_block_contrib = [
        {0:1, 3:1, 4:-1, 6:1},    # C00
        {2:1, 4:1},                # C01
        {1:1, 3:1},                # C10
        {0:1, 1:-1, 2:1, 5:1},    # C11
    ]
    
    # For each outer product s and inner product t (0-6), and for each of the two B-halves (0,1):
    # Column = s*7 + t + half*49
    
    for half in range(2):
        for s in range(7):
            a_coeff_dict, b_coeff_dict = outer_products[s]
            for t in range(7):
                col = half * 49 + s * 7 + t
                # Get U22, V22, W22 for inner product t
                # Add A coefficients
                for blk, coeff_a in a_coeff_dict.items():
                    for elem in a_block_elems[blk]:
                        for local_i in range(4):
                            if U22[local_i][t] != 0:
                                U[elem][col] += coeff_a * U22[local_i][t]
                # Add B coefficients
                b_offset = half * 16  # B1 starts at 0, B2 starts at 16 in V
                for blk, coeff_b in b_coeff_dict.items():
                    for elem in a_block_elems[blk]:  # Same block structure for B
                        for local_i in range(4):
                            if V22[local_i][t] != 0:
                                V[b_offset + elem][col] += coeff_b * V22[local_i][t]
                # Add W coefficients
                for c_blk, coeff_c in c_block_contrib[s].items():
                    w_offset = half * 16 + c_blk * 4  # C-block offset
                    for local_i in range(4):
                        if W22[local_i][t] != 0:
                            W[w_offset + local_i][col] += coeff_c * W22[local_i][t]
    
    return {"U": U, "V": V, "W": W}
