import numpy as np

def construct():
    n, m, p = 4, 4, 9
    
    # Dimensions
    dim_U = n * m  # 16
    dim_V = m * p  # 36
    dim_W = p * n  # 36
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
        
    # Strassen for 2x2 * 2x2
    # A = [[a, b], [c, d]]
    # B = [[e, f], [g, h]]
    # C = [[c11, c12], [c21, c22]]
    #
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    #
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # We represent each term as (u_vec, v_vec, w_vec)
    # u_vec has 4 entries corresponding to a, b, c, d
    # v_vec has 4 entries corresponding to e, f, g, h
    # w_vec has 4 entries corresponding to c11, c12, c21, c22
    
    # Note: The tensor decomposition sums $u \cdot v \cdot w$.
    # So for a term like m1 = (a+d)(e+h), the contribution to c11 is +1, to c22 is +1.
    # The term is:
    # u = [1, 0, 0, 1] (a+d)
    # v = [1, 0, 0, 1] (e+h)
    # w = [1, 0, 0, 1] (c11 + c22)
    # This single rank-1 term adds (a+d)(e+h) to c11 and (a+d)(e+h) to c22.
    
    strassen_2x2_terms = [
        # m1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # m2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # m3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # m4
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
        # m5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # m6
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]),
        # m7
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),
    ]
    
    def add_strassen_2x2_block(A_offset_i, A_offset_j, B_offset_j, B_offset_k, C_offset_i, C_offset_k):
        """
        Add Strassen terms for a 2x2 * 2x2 block.
        A is at A[A_offset_i:A_offset_i+2, A_offset_j:A_offset_j+2]
        B is at B[B_offset_j:B_offset_j+2, B_offset_k:B_offset_k+2]
        C is at C[C_offset_i:C_offset_i+2, C_offset_k:C_offset_k+2]
        """
        for u_local, v_local, w_local in strassen_2x2_terms:
            u = [0] * dim_U
            v = [0] * dim_V
            w = [0] * dim_W
            
            # Map local A indices (0,1,2,3) to global
            # local 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
            for li in range(4):
                coeff = u_local[li]
                if coeff != 0:
                    li_row = li // 2
                    li_col = li % 2
                    gi = A_offset_i + li_row
                    gj = A_offset_j + li_col
                    u[gi * m + gj] = coeff
            
            # Map local B indices (0,1,2,3) to global
            # local 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
            for lj in range(4):
                coeff = v_local[lj]
                if coeff != 0:
                    lj_row = lj // 2
                    lj_col = lj % 2
                    gj = B_offset_j + lj_row
                    gk = B_offset_k + lj_col
                    v[gj * p + gk] = coeff
            
            # Map local C indices (0,1,2,3) to global
            # local 0 -> (0,0), 1 -> (0,1), 2 -> (1,0), 3 -> (1,1)
            for lk in range(4):
                coeff = w_local[lk]
                if coeff != 0:
                    lk_row = lk // 2
                    lk_col = lk % 2
                    gi = C_offset_i + lk_row
                    gk = C_offset_k + lk_col
                    # W index is gk * n + gi
                    w[gk * n + gi] = coeff
            
            add_term(u, v, w)
            
    def add_strassen_4x4_block(A_offset_i, A_offset_j, B_offset_j, B_offset_k, C_offset_i, C_offset_k):
        """
        Add Strassen terms for a 4x4 * 4x4 block.
        Uses 7 multiplications of 2x2 blocks.
        """
        # Block definitions for A, B, C
        # A11: (0,0), A12: (0,2), A21: (2,0), A22: (2,2) relative to offset
        
        # We need to define the 7 Strassen multiplications M1..M7
        # Each Mi is a product of linear combinations of A-blocks and B-blocks.
        # We express each Mi as a sum of products of 2x2 blocks?
        # No, each Mi is a single 2x2 * 2x2 multiplication.
        # The "A side" of Mi is a 2x2 matrix formed by summing A-blocks.
        # The "B side" of Mi is a 2x2 matrix formed by summing B-blocks.
        
        # We can reuse add_strassen_2x2_block if we can pass the "summed" blocks.
        # But add_strassen_2x2_block assumes specific offsets.
        # Instead, we will construct the u, v, w vectors for each of the 7 Strassen terms
        # of the 4x4 problem directly.
        
        # Let's define the 7 block multiplications.
        # A_blocks:
        # A11 at (0,0), A12 at (0,2), A21 at (2,0), A22 at (2,2)
        # B_blocks:
        # B11 at (0,0), B12 at (0,2), B21 at (2,0), B22 at (2,2)
        # C_blocks:
        # C11 at (0,0), C12 at (0,2), C21 at (2,0), C22 at (2,2)
        
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We will iterate through the 7 Strassen terms of the 2x2 meta-matrix.
        # For each meta-term, we have coefficients for A' entries and B' entries.
        # A' entries are sums of A-block entries.
        
        # Let's define a helper to add a term given:
        # A_side: list of (block_row, block_col, coeff)
        # B_side: list of (block_row, block_col, coeff)
        # C_side: list of (block_row, block_col, coeff)
        
        def add_meta_term(A_side, B_side, C_side):
            u = [0] * dim_U
            v = [0] * dim_V
            w = [0] * dim_W
            
            # We need to expand the 2x2 Strassen terms inside this meta-term?
            # No, this function is called for each of the 7 Strassen terms of the 2x2 meta-matrix.
            # But wait, the meta-matrix multiplication itself requires 7 multiplications.
            # So we need to call the 2x2 Strassen decomposition on the meta-matrix.
            
            # This approach is getting nested. Let's flatten it.
            # Total 49 terms.
            # Term (i, j) where i in 0..6 (meta-Strassen), j in 0..6 (inner-Strassen).
            
            pass # This helper is not used directly in the flattened approach below.
            
        # Flattened approach:
        # Define the 7 meta-Strassen terms for the 2x2 block multiplication.
        # meta_terms[i] = (A_coeffs, B_coeffs, C_coeffs)
        # A_coeffs: dict (local_row, local_col) -> coeff
        # B_coeffs: dict (local_row, local_col) -> coeff
        # C_coeffs: dict (local_row, local_col) -> coeff
        
        meta_terms = [
            # s1 = (a+d)(e+h)
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # s2 = (c+d)e
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            # s3 = a(f-h)
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            # s4 = d(g-e)
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # s5 = (a+b)h
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
            # s6 = (c-a)(e+f)
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # s7 = (b-d)(g+h)
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        # Define the 7 block multiplications M1..M7
        # Each M is defined by A_side, B_side, C_side
        # A_side: list of (block_row, block_col, coeff)
        # B_side: list of (block_row, block_col, coeff)
        # C_side: list of (block_row, block_col, coeff)
        
        block_mults = [
            # M1
            ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
            # M2
            ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)]),
            # M3
            ([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)]),
            # M4
            ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)]),
            # M5
            ([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)]),
            # M6
            ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
            # M7
            ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]),
        ]
        
        for A_side, B_side, C_side in block_mults:
            for A_coeffs, B_coeffs, C_coeffs in meta_terms:
                u = [0] * dim_U
                v = [0] * dim_V
                w = [0] * dim_W
                
                # Expand A
                for (lr, lc), meta_coeff in A_coeffs.items():
                    for (br, bc, block_coeff) in A_side:
                        # Global A index
                        gi = A_offset_i + br * 2 + lr
                        gj = A_offset_j + bc * 2 + lc
                        u[gi * m + gj] += meta_coeff * block_coeff
                        
                # Expand B
                for (lr, lc), meta_coeff in B_coeffs.items():
                    for (br, bc, block_coeff) in B_side:
                        # Global B index
                        gj = B_offset_j + br * 2 + lr
                        gk = B_offset_k + bc * 2 + lc
                        v[gj * p + gk] += meta_coeff * block_coeff
                        
                # Expand C
                for (lr, lc), meta_coeff in C_coeffs.items():
                    for (br, bc, block_coeff) in C_side:
                        # Global C index
                        gi = C_offset_i + br * 2 + lr
                        gk = C_offset_k + bc * 2 + lc
                        w[gk * n + gi] += meta_coeff * block_coeff
                        
                add_term(u, v, w)

    # Part 1: A * B1 (cols 0-3)
    add_strassen_4x4_block(0, 0, 0, 0, 0, 0)
    
    # Part 2: A * B2 (cols 4-7)
    add_strassen_4x4_block(0, 0, 0, 4, 0, 4)
    
    # Part 3: A * B3 (col 8)
    # Schoolbook for 4x4 * 4x1
    for i in range(4):
        for j in range(4):
            u = [0] * dim_U
            v = [0] * dim_V
            w = [0] * dim_W
            u[i * m + j] = 1
            v[j * p + 8] = 1
            w[8 * n + i] = 1
            add_term(u, v, w)
            
    # Construct matrices
    rank = len(U_cols)
    U = np.array(U_cols).T  # Shape (16, rank)
    V = np.array(V_cols).T  # Shape (36, rank)
    W = np.array(W_cols).T  # Shape (36, rank)
    
    return {"U": U, "V": V, "W": W}
