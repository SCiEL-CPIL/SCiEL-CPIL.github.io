"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    # We'll build a decomposition of rank 93:
    # Part 1: 4x4x4 via Strassen^2 (rank 49)
    # Part 2: 4x4x2 via 4 independent Strassen 2x2x2 (rank 28)
    # Part 3: 4x4x1 via schoolbook (rank 16)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper: Strassen 2x2x2 decomposition
    # Returns U_vec (4,), V_vec (4,), W_vec (4,) for each of the 7 products
    def strassen_2x2x2():
        # A indices: 0=A00, 1=A01, 2=A10, 3=A11
        # B indices: 0=B00, 1=B01, 2=B10, 3=B11
        # C indices: 0=C00, 1=C01, 2=C10, 3=C11
        products = [
            # (U_vec, V_vec, W_vec)
            ([1,0,0,1], [1,0,0,1], [1,0,0,1]),   # p1
            ([0,0,1,1], [1,0,0,0], [0,0,1,-1]),   # p2
            ([1,0,0,0], [0,1,0,-1], [0,1,0,1]),   # p3
            ([0,0,0,1], [0,0,1,-1], [1,0,1,0]),   # p4
            ([1,1,0,0], [0,0,0,1], [-1,1,0,0]),   # p5
            ([-1,0,1,0], [1,1,0,0], [0,0,0,1]),   # p6
            ([0,1,0,-1], [0,0,1,1], [1,0,0,0]),   # p7
        ]
        return products
    
    # Part 1: 4x4x4 using Strassen^2
    # A blocks: A11 (rows 0-1, cols 0-1), A12 (rows 0-1, cols 2-3), A21 (rows 2-3, cols 0-1), A22 (rows 2-3, cols 2-3)
    # B blocks: B11, B12, B21, B22 similarly
    # C blocks: C11, C12, C21, C22
    
    strassen_prods = strassen_2x2x2()
    
    # Block-level Strassen:
    # S1 = (A11+A22)(B11+B22)
    # S2 = (A21+A22)B11
    # S3 = A11(B12-B22)
    # S4 = A22(B21-B11)
    # S5 = (A11+A12)B22
    # S6 = (A21-A11)(B11+B12)
    # S7 = (A12-A22)(B21+B22)
    
    # For each block-level product, we need to combine the 2x2x2 Strassen decompositions
    # A11, A12, A21, A22 are 2x2 matrices, indexed in A as:
    # A11: A[0,0],A[0,1],A[1,0],A[1,1] -> indices 0,1,4,5 in 4x4 row-major
    # A12: A[0,2],A[0,3],A[1,2],A[1,3] -> indices 2,3,6,7
    # A21: A[2,0],A[2,1],A[3,0],A[3,1] -> indices 8,9,12,13
    # A22: A[2,2],A[2,3],A[3,2],A[3,3] -> indices 10,11,14,15
    
    # B11: B[0,0],B[0,1],B[1,0],B[1,1] -> indices 0,1,4,5
    # B12: B[0,2],B[0,3],B[1,2],B[1,3] -> indices 2,3,6,7
    # B21: B[2,0],B[2,1],B[3,0],B[3,1] -> indices 8,9,12,13
    # B22: B[2,2],B[2,3],B[3,2],B[3,3] -> indices 10,11,14,15
    
    # C11: C[0,0],C[0,1],C[1,0],C[1,1] -> W indices 0,4,1,5 (W[k*n+i] = k*4+i)
    # C12: C[0,2],C[0,3],C[1,2],C[1,3] -> W indices 8,9,10,11
    # C21: C[2,0],C[2,1],C[3,0],C[3,1] -> W indices 16,17,18,19
    # C22: C[2,2],C[2,3],C[3,2],C[3,3] -> W indices 20,21,22,23
    
    # For a 2x2x2 sub-problem with A-block at row-offset ra, col-offset ca (in 4x4 A),
    # B-block at row-offset rb, col-offset cb (in 4x4 B),
    # C-block at row-offset rc, col-offset cc (in 4x4 C, which maps to W)
    
    def add_strassen_2x2x2(U_full, V_full, W_full, a_row_off, a_col_off, b_row_off, b_col_off, c_row_off, c_col_off, scale=1):
        """Add a Strassen 2x2x2 decomposition for blocks at given offsets."""
        prods = strassen_2x2x2()
        for u_vec, v_vec, w_vec in prods:
            U_col = [0.0] * 16
            V_col = [0.0] * 28
            W_col = [0.0] * 28
            
            # A block: 2x2 at (a_row_off, a_col_off)
            # Local indices 0,1,2,3 -> global A indices
            a_local_to_global = [
                a_row_off * 4 + a_col_off,       # 0
                a_row_off * 4 + a_col_off + 1,   # 1
                (a_row_off+1) * 4 + a_col_off,   # 2
                (a_row_off+1) * 4 + a_col_off + 1, # 3
            ]
            for li, gi in enumerate(a_local_to_global):
                U_col[gi] = scale * u_vec[li]
            
            # B block: 2x2 at (b_row_off, b_col_off) in 4x4 B -> V index = b_row*4 + b_col
            b_local_to_global = [
                b_row_off * 4 + b_col_off,
                b_row_off * 4 + b_col_off + 1,
                (b_row_off+1) * 4 + b_col_off,
                (b_row_off+1) * 4 + b_col_off + 1,
            ]
            for li, gi in enumerate(b_local_to_global):
                V_col[gi] = scale * v_vec[li]
            
            # C block: 2x2 at (c_row_off, c_col_off) in 4x4 C -> W index = c_col*4 + c_row
            c_local_to_global = [
                c_col_off * 4 + c_row_off,
                c_col_off * 4 + c_row_off + 1,
                (c_col_off+1) * 4 + c_row_off,
                (c_col_off+1) * 4 + c_row_off + 1,
            ]
            for li, gi in enumerate(c_local_to_global):
                W_col[gi] = scale * w_vec[li]
            
            U_full.append(U_col)
            V_full.append(V_col)
            W_full.append(W_col)
    
    # S1 = (A11+A22)(B11+B22)
    # A11+A22: need to add two 2x2x2 decompositions
    # (A11+A22) * B11: A block is A11+A22, B block is B11
    # We can't directly add matrices in the tensor, but we can do:
    # (A11+A22)*B11 = A11*B11 + A22*B11
    # So S1 contributes: A11*B11 + A22*B11 + A11*B22 + A22*B22
    # Each is a 2x2x2 multiplication
    
    # S1: A11*B11, A22*B11, A11*B22, A22*B22
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 0, 0, 0, 0, 0, 0, 1)    # A11*B11 -> C11
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 2, 2, 0, 0, 0, 0, 1)    # A22*B11 -> C11
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 0, 0, 2, 2, 0, 0, 1)    # A11*B22 -> C11
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 2, 2, 2, 2, 0, 0, 1)    # A22*B22 -> C11
    
    # But wait, S1 = (A11+A22)(B11+B22) = A11*B11 + A11*B22 + A22*B11 + A22*B22
    # And C11 gets +1 * S1
    # So we need C11 to get the sum of these 4 products.
    # Each add_strassen_2x2x2 adds 7 rank-1 terms that contribute to the C block.
    # So C11 will get contributions from all 4 calls above. Good.
    
    # S2 = (A21+A22)B11 = A21*B11 + A22*B11
    # C21 gets +1 * S2
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 2, 0, 0, 0, 2, 0, 1)    # A21*B11 -> C21
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 2, 2, 0, 0, 2, 0, 1)    # A22*B11 -> C21
    
    # S3 = A11(B12-B22) = A11*B12 - A11*B22
    # C12 gets +1 * S3
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 0, 0, 0, 2, 0, 2, 1)    # A11*B12 -> C12
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 0, 0, 2, 2, 0, 2, -1)   # A11*B22 -> C12 (with -1)
    
    # S4 = A22(B21-B11) = A22*B21 - A22*B11
    # C11 gets +1 * S4, C21 gets +1 * S4
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 2, 2, 2, 0, 0, 0, 1)    # A22*B21 -> C11
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 2, 2, 0, 0, 0, 0, -1)   # A22*B11 -> C11 (with -1)
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 2, 2, 2, 0, 2, 0, 1)    # A22*B21 -> C21
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 2, 2, 0, 0, 2, 0, -1)   # A22*B11 -> C21 (with -1)
    
    # S5 = (A11+A12)B22 = A11*B22 + A12*B22
    # C11 gets -1 * S5, C12 gets +1 * S5
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 0, 0, 2, 2, 0, 0, -1)   # A11*B22 -> C11
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 0, 2, 2, 2, 0, 0, -1)   # A12*B22 -> C11
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 0, 0, 2, 2, 0, 2, 1)    # A11*B22 -> C12
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 0, 2, 2, 2, 0, 2, 1)    # A12*B22 -> C12
    
    # S6 = (A21-A11)(B11+B12) = A21*B11 + A21*B12 - A11*B11 - A11*B12
    # C22 gets +1 * S6
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 2, 0, 0, 0, 2, 2, 1)    # A21*B11 -> C22
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 2, 0, 0, 2, 2, 2, 1)    # A21*B12 -> C22
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 0, 0, 0, 0, 2, 2, -1)   # A11*B11 -> C22
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 0, 0, 0, 2, 2, 2, -1)   # A11*B12 -> C22
    
    # S7 = (A12-A22)(B21+B22) = A12*B21 + A12*B22 - A22*B21 - A22*B22
    # C11 gets +1 * S7
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 0, 2, 2, 0, 0, 0, 1)    # A12*B21 -> C11
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 0, 2, 2, 2, 0, 0, 1)    # A12*B22 -> C11
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 2, 2, 2, 0, 0, 0, -1)   # A22*B21 -> C11
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 2, 2, 2, 2, 0, 0, -1)   # A22*B22 -> C11
    
    # Part 1 done: 49 terms (7 Strassen products * 7 each)
    
    # Part 2: 4x4x2 (columns 4-5 of B and C)
    # B columns 4-5: B1 = B[0:2, 4:6], B2 = B[2:4, 4:6]
    # C columns 4-5: C_top = C[0:2, 4:6], C_bottom = C[2:4, 4:6]
    # C_top = A11*B1 + A12*B2
    # C_bottom = A21*B1 + A22*B2
    
    # A11*B1 -> C_top
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 0, 0, 0, 4, 0, 4, 1)
    # A12*B2 -> C_top
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 0, 2, 2, 4, 0, 4, 1)
    # A21*B1 -> C_bottom
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 2, 0, 0, 4, 2, 4, 1)
    # A22*B2 -> C_bottom
    add_strassen_2x2x2(U_cols, V_cols, W_cols, 2, 2, 2, 4, 2, 4, 1)
    
    # Part 2 done: 28 terms
    
    # Part 3: 4x4x1 (column 6 of B and C)
    # Schoolbook: 16 terms
    for i in range(4):
        for j in range(4):
            U_col = [0.0] * 16
            V_col = [0.0] * 28
            W_col = [0.0] * 28
            U_col[i * 4 + j] = 1
            V_col[j * 4 + 6] = 1  # B[j, 6]
            W_col[6 * 4 + i] = 1  # C[i, 6] -> W[6*4+i]
            U_cols.append(U_col)
            V_cols.append(V_col)
            W_cols.append(W_col)
    
    # Part 3 done: 16 terms
    
    # Total rank: 49 + 28 + 16 = 93
    rank = len(U_cols)
    U = U_cols
    V = V_cols
    W = W_cols
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
