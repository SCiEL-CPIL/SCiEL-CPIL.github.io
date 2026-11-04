import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank)
    # V: (m*p, rank)
    # W: (p*n, rank)
    
    # We will build U, V, W as lists of columns
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_strassen_2x2(offset_A, offset_B, offset_C):
        # Adds 7 columns to U_cols, V_cols, W_cols
        # Local indices 0..3 for 2x2 blocks
        # A: 0:00, 1:01, 2:10, 3:11
        # B: 0:00, 1:01, 2:10, 3:11
        # C: 0:00, 1:01, 2:10, 3:11
        
        # Define the 7 terms (u, v, w_contribs)
        # w_contribs is a dict {local_c_index: coeff}
        
        # M1: (A00+A11)(B00+B11)
        u1 = [0]*4; u1[0]=1; u1[3]=1
        v1 = [0]*4; v1[0]=1; v1[3]=1
        w1 = {0:1, 3:1}
        
        # M2: (A10+A11)B00
        u2 = [0]*4; u2[2]=1; u2[3]=1
        v2 = [0]*4; v2[0]=1
        w2 = {2:1, 3:-1}
        
        # M3: A00(B01-B11)
        u3 = [0]*4; u3[0]=1
        v3 = [0]*4; v3[1]=1; v3[3]=-1
        w3 = {1:1, 3:1}
        
        # M4: A11(B10-B00)
        u4 = [0]*4; u4[3]=1
        v4 = [0]*4; v4[2]=1; v4[0]=-1
        w4 = {0:1, 2:1}
        
        # M5: (A00+A01)B11
        u5 = [0]*4; u5[0]=1; u5[1]=1
        v5 = [0]*4; v5[3]=1
        w5 = {0:-1, 1:1}
        
        # M6: (A10-A00)(B00+B01)
        u6 = [0]*4; u6[2]=1; u6[0]=-1
        v6 = [0]*4; v6[0]=1; v6[1]=1
        w6 = {3:1}
        
        # M7: (A01-A11)(B10+B11)
        u7 = [0]*4; u7[1]=1; u7[3]=-1
        v7 = [0]*4; v7[2]=1; v7[3]=1
        w7 = {0:1}
        
        terms = [
            (u1, v1, w1),
            (u2, v2, w2),
            (u3, v3, w3),
            (u4, v4, w4),
            (u5, v5, w5),
            (u6, v6, w6),
            (u7, v7, w7)
        ]
        
        for u, v, w in terms:
            # Expand to global sizes
            # U size: n*m = 16
            # V size: m*p = 24
            # W size: p*n = 24
            
            U_vec = [0.0] * (n*m)
            V_vec = [0.0] * (m*p)
            W_vec = [0.0] * (p*n)
            
            for i in range(4):
                if u[i] != 0:
                    U_vec[offset_A + i] = u[i]
                if v[i] != 0:
                    V_vec[offset_B + i] = v[i]
                if i in w:
                    W_vec[offset_C + i] = w[i]
            
            U_cols.append(U_vec)
            V_cols.append(V_vec)
            W_cols.append(W_vec)
            
    # Part 1: 4x4 * 4x4 (Rank 49)
    # A is 4x4, B1 is 4x4 (first 4 cols of B), C1 is 4x4 (first 4 cols of C)
    # Offsets:
    # A: 0
    # B: 0 (B is 4x6, B1 is cols 0-3, so indices 0-15)
    # C: 0 (C is 4x6, C1 is cols 0-3, so indices 0-15)
    
    # Strassen for 4x4x4 uses 7 calls to 2x2x2 Strassen
    # A blocks: A11(0), A12(4), A21(8), A22(12)
    # B blocks: B11(0), B12(4), B21(8), B22(12)
    # C blocks: C11(0), C12(4), C21(8), C22(12)
    
    # M1: (A11+A22)(B11+B22) -> C11, C22
    # This is a 2x2 multiplication of the block matrices.
    # We call add_strassen_2x2 7 times.
    # But the "entries" of the block matrix are 2x2 matrices.
    # So each "entry" multiplication is a 2x2x2 Strassen.
    
    # The 7 Strassen products for 4x4:
    # S1 = (A11+A22)(B11+B22)
    # S2 = (A21+A22)B11
    # S3 = A11(B12-B22)
    # S4 = A22(B21-B11)
    # S5 = (A11+A12)B22
    # S6 = (A21-A11)(B11+B12)
    # S7 = (A12-A22)(B21+B22)
    
    # Each Si is a 2x2 matrix product.
    # We need to decompose each Si into 7 rank-1 tensors.
    
    # Helper to call add_strassen_2x2 for a specific block product
    # Note: add_strassen_2x2 handles the 7 terms for a 2x2x2 multiplication.
    # Here we are doing 7 such multiplications.
    
    # S1: A11+A22 times B11+B22
    # This is not a single 2x2 multiplication in the Strassen sense of the 4x4 algorithm?
    # Yes, it is. The "elements" are 2x2 matrices.
    # So S1 = (A11+A22) * (B11+B22).
    # This requires 7 multiplications of 2x2 matrices?
    # No. S1 is ONE product of two 2x2 matrices (which are sums).
    # To compute S1, we need to multiply (A11+A22) by (B11+B22).
    # This is a 2x2x2 multiplication.
    # So we call add_strassen_2x2 once for this?
    # No, add_strassen_2x2 computes a full 2x2x2 multiplication (7 terms).
    # So for S1, we call add_strassen_2x2 with offsets for A11+A22?
    # No, add_strassen_2x2 assumes the inputs are single 2x2 blocks.
    # It doesn't handle sums of blocks in the U/V vectors automatically.
    # Wait, my add_strassen_2x2 implementation above takes offsets for A, B, C.
    # It assumes A is a single 2x2 block.
    # But in Strassen 4x4, the inputs to the 7 products are sums of blocks.
    # E.g., A11+A22.
    # So I need to modify add_strassen_2x2 or create a new function that takes a list of (offset, coeff) for A and B.
    
    def add_strassen_2x2_general(A_terms, B_terms, C_terms):
        # A_terms: list of (offset, coeff)
        # B_terms: list of (offset, coeff)
        # C_terms: list of (offset, coeff)
        # This adds 7 columns corresponding to the 7 Strassen products of the linear combos.
        
        # We need to generate the 7 u, v, w vectors for the Strassen 2x2x2.
        # Then for each, we construct the global vectors by summing contributions from A_terms and B_terms.
        
        # Local u, v, w patterns for 2x2x2 Strassen
        # u1: [1,0,0,1], v1: [1,0,0,1], w1: {0:1, 3:1}
        # ...
        
        patterns = [
            ([1,0,0,1], [1,0,0,1], {0:1, 3:1}),
            ([0,0,1,1], [1,0,0,0], {2:1, 3:-1}),
            ([1,0,0,0], [0,1,-1,0], {1:1, 3:1}),
            ([0,0,0,1], [-1,0,1,0], {0:1, 2:1}),
            ([1,1,0,0], [0,0,0,1], {0:-1, 1:1}),
            ([-1,0,1,0], [1,1,0,0], {3:1}),
            ([0,1,0,-1], [0,0,1,1], {0:1})
        ]
        
        for u_loc, v_loc, w_loc in patterns:
            U_vec = [0.0] * (n*m)
            V_vec = [0.0] * (m*p)
            W_vec = [0.0] * (p*n)
            
            # Construct U_vec
            for i in range(4):
                if u_loc[i] != 0:
                    for off, coeff in A_terms:
                        U_vec[off + i] += u_loc[i] * coeff
            
            # Construct V_vec
            for i in range(4):
                if v_loc[i] != 0:
                    for off, coeff in B_terms:
                        V_vec[off + i] += v_loc[i] * coeff
            
            # Construct W_vec
            for i in range(4):
                if i in w_loc:
                    for off, coeff in C_terms:
                        W_vec[off + i] += w_loc[i] * coeff
            
            U_cols.append(U_vec)
            V_cols.append(V_vec)
            W_cols.append(W_vec)

    # Define block offsets
    # A: 4x4
    A11 = 0; A12 = 4; A21 = 8; A22 = 12
    # B: 4x6. B1 is 4x4 (cols 0-3), B2 is 4x2 (cols 4-5)
    # B1 blocks:
    B11 = 0; B12 = 4; B21 = 8; B22 = 12
    # B2 blocks (4x2):
    # B2 is rows 0-3, cols 4-5.
    # Flattened B:
    # Row 0: 0,1,2,3,4,5
    # Row 1: 6,7,8,9,10,11
    # Row 2: 12,13,14,15,16,17
    # Row 3: 18,19,20,21,22,23
    # B1 (cols 0-3): indices 0-3, 6-9, 12-15, 18-21
    # B2 (cols 4-5): indices 4-5, 10-11, 16-17, 22-23
    
    # Wait, my previous assumption about B2 being contiguous 4x2 was wrong for row-major flattening.
    # B is 4x6.
    # B2 is the last 2 columns.
    # Top 2x2 of B2: Rows 0-1, Cols 4-5.
    # Indices:
    # R0C4=4, R0C5=5
    # R1C4=10, R1C5=11
    # So B2_top = [4,5,10,11]
    # B2_bot = Rows 2-3, Cols 4-5.
    # R2C4=16, R2C5=17
    # R3C4=22, R3C5=23
    # So B2_bot = [16,17,22,23]
    
    # C: 4x6. C1 is 4x4 (cols 0-3), C2 is 4x2 (cols 4-5)
    # C1 blocks:
    C11 = 0; C12 = 4; C21 = 8; C22 = 12
    # C2 blocks:
    C2_top = [4,5,10,11]
    C2_bot = [16,17,22,23]

    # Part 1: 4x4 * 4x4 (B1)
    # S1: (A11+A22)(B11+B22)
    add_strassen_2x2_general(
        [(A11, 1), (A22, 1)],
        [(B11, 1), (B22, 1)],
        [(C11, 1), (C22, 1)]
    )
    # S2: (A21+A22)B11
    add_strassen_2x2_general(
        [(A21, 1), (A22, 1)],
        [(B11, 1)],
        [(C21, 1), (C22, -1)]
    )
    # S3: A11(B12-B22)
    add_strassen_2x2_general(
        [(A11, 1)],
        [(B12, 1), (B22, -1)],
        [(C12, 1), (C22, 1)]
    )
    # S4: A22(B21-B11)
    add_strassen_2x2_general(
        [(A22, 1)],
        [(B21, 1), (B11, -1)],
        [(C11, 1), (C21, 1)]
    )
    # S5: (A11+A12)B22
    add_strassen_2x2_general(
        [(A11, 1), (A12, 1)],
        [(B22, 1)],
        [(C11, -1), (C12, 1)]
    )
    # S6: (A21-A11)(B11+B12)
    add_strassen_2x2_general(
        [(A21, 1), (A11, -1)],
        [(B11, 1), (B12, 1)],
        [(C22, 1)]
    )
    # S7: (A12-A22)(B21+B22)
    add_strassen_2x2_general(
        [(A12, 1), (A22, -1)],
        [(B21, 1), (B22, 1)],
        [(C11, 1)]
    )

    # Part 2: 4x4 * 4x2 (B2)
    # C1 = A11*B2_top + A12*B2_bot
    # C2 = A21*B2_top + A22*B2_bot
    
    # Term 1: A11 * B2_top -> C1_top (which is C2_top? No, C1 is the 4x4 part. C2 is the 4x2 part.)
    # Wait, C is 4x6.
    # C1 (cols 0-3) is 4x4.
    # C2 (cols 4-5) is 4x2.
    # C2_top is rows 0-1 of C2.
    # C2_bot is rows 2-3 of C2.
    
    # A11 * B2_top contributes to C2_top?
    # C2_top = (A11 * B2_top + A12 * B2_bot)
    # A11 is top-left 2x2 of A.
    # B2_top is top 2x2 of B2.
    # Product is 2x2.
    # It adds to C2_top.
    
    add_strassen_2x2_general(
        [(A11, 1)],
        [(B2_top[0], 1), (B2_top[1], 1), (B2_top[2], 1), (B2_top[3], 1)], # Wait, B2_top is not contiguous.
        [(C2_top[0], 1), (C2_top[1], 1), (C2_top[2], 1), (C2_top[3], 1)]
    )
    # This is wrong. add_strassen_2x2_general expects offsets for 2x2 blocks.
    # But B2_top is not a contiguous 2x2 block in the flattened array.
    # My add_strassen_2x2_general adds `off + i`.
    # This assumes the 2x2 block is contiguous.
    # For B2_top, indices are 4,5,10,11.
    # 4+0=4, 4+1=5, 4+2=6 (Wrong, should be 10), 4+3=7 (Wrong, should be 11).
    
    # I need to handle non-contiguous blocks.
    # I will modify add_strassen_2x2_general to take explicit index lists for A, B, C.
    
    # Let's redefine add_strassen_2x2_general to take lists of 4 indices for A, B, C.
    
    # I'll rewrite the logic in the final code.

