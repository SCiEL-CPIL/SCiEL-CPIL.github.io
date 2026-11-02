def construct():
    n, m, p = 4, 4, 9
    
    # We'll build the decomposition for A(4x4) * B(4x9)
    # Partition B into B1(4x4), B2(4x4), B3(4x1)
    # Use Strassen for A*B1 and A*B2 (rank 49 each)
    # Use schoolbook for A*B3 (rank 16)
    # Total rank: 49 + 49 + 16 = 114
    
    rank = 114
    U = [[0] * rank for _ in range(n * m)]  # (16, 114)
    V = [[0] * rank for _ in range(m * p)]  # (36, 114)
    W = [[0] * rank for _ in range(p * n)]  # (36, 114)
    
    col = 0
    
    # Helper: add a Strassen 4x4x4 decomposition starting at column col
    # Returns new column index
    def add_strassen_4x4x4(col_offset, B_col_offset):
        """
        Add Strassen decomposition for A(4x4) * B(4x4) where B starts at column B_col_offset.
        Returns col_offset + 49.
        """
        # Top-level Strassen: 7 multiplications of 2x2 matrices
        # Each 2x2 multiplication uses Strassen (7 scalar mults)
        
        # Define the 7 top-level multiplications
        # Each is: (A_lin) * (B_lin) where A_lin and B_lin are 2x2 matrices
        # formed by linear combinations of A and B blocks.
        
        # A blocks: A00=(0,0;0,1), A01=(0,2;0,3), A10=(2,0;2,1), A11=(2,2;2,3)
        # B blocks: B00=(0,0;0,1), B01=(0,2;0,3), B10=(2,0;2,1), B11=(2,2;2,3)
        
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # For each M, the 2x2 result contributes to C blocks:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # Define the 7 top-level multiplications as (A_block_coeffs, B_block_coeffs)
        # where each block is 2x2, and coeffs are +1, -1, or 0
        
        # A blocks indexed: 0=A00, 1=A01, 2=A10, 3=A11
        # B blocks indexed: 0=B00, 1=B01, 2=B10, 3=B11
        
        top_mults = [
            # (A_block_coeffs, B_block_coeffs)
            ([1, 0, 0, 1], [1, 0, 0, 1]),  # M1: (A00+A11)(B00+B11)
            ([0, 0, 1, 1], [1, 0, 0, 0]),  # M2: (A10+A11)B00
            ([1, 0, 0, 0], [0, 1, 0, -1]),  # M3: A00(B01-B11)
            ([0, 0, 0, 1], [0, 0, 1, -1]),  # M4: A11(B10-B00)
            ([1, 1, 0, 0], [0, 0, 0, 1]),  # M5: (A00+A01)B11
            ([0, 0, 1, -1], [1, 1, 0, 0]),  # M6: (A10-A00)(B00+B01)
            ([0, 1, 0, -1], [0, 0, 1, 1]),  # M7: (A01-A11)(B10+B11)
        ]
        
        # C block contributions for each M:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # C blocks: 0=C00, 1=C01, 2=C10, 3=C11
        c_contribs = [
            # For M1: C00(+1), C11(+1)
            {0: 1, 3: 1},
            # For M2: C10(+1), C11(-1)
            {2: 1, 3: -1},
            # For M3: C01(+1), C11(+1)
            {1: 1, 3: 1},
            # For M4: C00(+1), C10(+1)
            {0: 1, 2: 1},
            # For M5: C00(-1), C01(+1)
            {0: -1, 1: 1},
            # For M6: C11(+1)
            {3: 1},
            # For M7: C00(+1)
            {0: 1},
        ]
        
        # For each top-level multiplication, apply 2x2 Strassen
        # 2x2 Strassen: 7 scalar multiplications
        # A' = [[a,b],[c,d]], B' = [[e,f],[g,h]]
        # M1' = (a+d)(e+h)
        # M2' = (c+d)e
        # M3' = a(f-h)
        # M4' = d(g-e)
        # M5' = (a+b)h
        # M6' = (c-a)(e+f)
        # M7' = (b-d)(g+h)
        # C00' = M1' + M4' - M5' + M7'
        # C01' = M3' + M5'
        # C10' = M2' + M4'
        # C11' = M1' - M2' + M3' + M6'
        
        # For the 2x2 Strassen, the 7 scalar mults are:
        # Each is: (linear combo of A') * (linear combo of B')
        # A' elements: (0,0)=a, (0,1)=b, (1,0)=c, (1,1)=d
        # B' elements: (0,0)=e, (0,1)=f, (1,0)=g, (1,1)=h
        
        strassen_2x2 = [
            # (A_coeffs, B_coeffs, C_contribs)
            # A_coeffs: [a, b, c, d] coefficients
            # B_coeffs: [e, f, g, h] coefficients
            # C_contribs: {C00': coeff, C01': coeff, C10': coeff, C11': coeff}
            ([1, 0, 0, 1], [1, 0, 0, 1], {0: 1, 3: 1}),  # (a+d)(e+h)
            ([0, 0, 1, 1], [1, 0, 0, 0], {2: 1, 3: -1}),  # (c+d)e
            ([1, 0, 0, 0], [0, 1, 0, -1], {1: 1, 3: 1}),  # a(f-h)
            ([0, 0, 0, 1], [0, 0, 1, -1], {0: 1, 2: 1}),  # d(g-e)
            ([1, 1, 0, 0], [0, 0, 0, 1], {0: -1, 1: 1}),  # (a+b)h
            ([0, 0, 1, -1], [1, 1, 0, 0], {3: 1}),  # (c-a)(e+f)
            ([0, 1, 0, -1], [0, 0, 1, 1], {0: 1}),  # (b-d)(g+h)
        ]
        
        for mi, (a_block_coeffs, b_block_coeffs) in enumerate(top_mults):
            # For this top-level multiplication, A' is a 2x2 matrix where
            # A'[r,s] = sum over A blocks: a_block_coeffs[block] * A[block_row+r, block_col+s]
            # Similarly for B'
            
            # For each of the 7 Strassen scalar mults:
            for (a_coeffs, b_coeffs, c2_contribs) in strassen_2x2:
                # a_coeffs: [a, b, c, d] for A' = [[a,b],[c,d]]
                # b_coeffs: [e, f, g, h] for B' = [[e,f],[g,h]]
                
                # Build U vector (length 16 for A)
                # U[i*m+j, col] = coefficient of A[i,j]
                u_vec = [0] * (n * m)  # 16
                v_vec = [0] * (m * 4)  # 16 for the 4x4 B part
                w_vec = [0] * (4 * n)  # 16 for the 4x4 C part
                
                # A' element (r,s) corresponds to A' positions:
                # A'[0,0] = sum over A blocks: a_block_coeffs[block] * A[block_r + 0, block_c + 0]
                # where block_r, block_c depend on the block index
                
                # A blocks:
                # Block 0 (A00): rows 0-1, cols 0-1
                # Block 1 (A01): rows 0-1, cols 2-3
                # Block 2 (A10): rows 2-3, cols 0-1
                # Block 3 (A11): rows 2-3, cols 2-3
                
                # For A'[r,s]:
                # A'[0,0] comes from A blocks at positions (0,0) within each block
                # A'[0,1] comes from A blocks at positions (0,1) within each block
                # A'[1,0] comes from A blocks at positions (1,0) within each block
                # A'[1,1] comes from A blocks at positions (1,1) within each block
                
                a_positions = [
                    # For A'[0,0]: (row, col) in A for each block
                    [(0,0), (0,2), (2,0), (2,2)],  # block 0,1,2,3 at position (0,0)
                    # For A'[0,1]:
                    [(0,1), (0,3), (2,1), (2,3)],
                    # For A'[1,0]:
                    [(1,0), (1,2), (3,0), (3,2)],
                    # For A'[1,1]:
                    [(1,1), (1,3), (3,1), (3,3)],
                ]
                
                # a_coeffs = [a, b, c, d] corresponds to A'[0,0], A'[0,1], A'[1,0], A'[1,1]
                for idx, coeff in enumerate(a_coeffs):
                    if coeff != 0:
                        for block_idx, (row, col) in enumerate(a_positions[idx]):
                            block_coeff = a_block_coeffs[block_idx]
                            if block_coeff != 0:
                                u_vec[row * m + col] += coeff * block_coeff
                
                # B' element positions:
                # B'[0,0]: B blocks at position (0,0) within each block
                # B'[0,1]: B blocks at position (0,1) within each block
                # B'[1,0]: B blocks at position (1,0) within each block
                # B'[1,1]: B blocks at position (1,1) within each block
                
                # B blocks:
                # Block 0 (B00): rows 0-1, cols 0-1
                # Block 1 (B01): rows 0-1, cols 2-3
                # Block 2 (B10): rows 2-3, cols 0-1
                # Block 3 (B11): rows 2-3, cols 2-3
                
                b_positions = [
                    [(0,0), (0,2), (2,0), (2,2)],
                    [(0,1), (0,3), (2,1), (2,3)],
                    [(1,0), (1,2), (3,0), (3,2)],
                    [(1,1), (1,3), (3,1), (3,3)],
                ]
                
                # b_coeffs = [e, f, g, h] corresponds to B'[0,0], B'[0,1], B'[1,0], B'[1,1]
                for idx, coeff in enumerate(b_coeffs):
                    if coeff != 0:
                        for block_idx, (row, col) in enumerate(b_positions[idx]):
                            block_coeff = b_block_coeffs[block_idx]
                            if block_coeff != 0:
                                # B is 4x4, so V index is row * 4 + col
                                # But in the full problem, B is 4x9, so we need to offset
                                # For B1 (first 4x4), offset is 0
                                # For B2 (second 4x4), offset is 4
                                v_vec[row * 4 + col + B_col_offset] += coeff * block_coeff
                
                # C' element positions:
                # C'[0,0]: C blocks at position (0,0)
                # C'[0,1]: C blocks at position (0,1)
                # C'[1,0]: C blocks at position (1,0)
                # C'[1,1]: C blocks at position (1,1)
                
                # C blocks:
                # Block 0 (C00): rows 0-1, cols 0-1
                # Block 1 (C01): rows 0-1, cols 2-3
                # Block 2 (C10): rows 2-3, cols 0-1
                # Block 3 (C11): rows 2-3, cols 2-3
                
                c_positions = [
                    [(0,0), (0,2), (2,0), (2,2)],
                    [(0,1), (0,3), (2,1), (2,3)],
                    [(1,0), (1,2), (3,0), (3,2)],
                    [(1,1), (1,3), (3,1), (3,3)],
                ]
                
                # c2_contribs: {C00': coeff, C01': coeff, C10': coeff, C11': coeff}
                # C' block 0 = C00, block 1 = C01, block 2 = C10, block 3 = C11
                for c2_idx, c2_coeff in c2_contribs.items():
                    if c2_coeff == 0:
                        continue
                    # c2_idx is the C' block index (0-3)
                    # But wait, c2_contribs uses keys 0,1,2,3 for C00', C01', C10', C11'
                    # And c_positions[c2_idx] gives the 4 positions in C for that block
                    
                    # Actually, I need to map c2_idx to the C block, then to positions
                    # c2_idx 0 -> C00 block, positions [(0,0),(0,1),(1,0),(1,1)]
                    # c2_idx 1 -> C01 block, positions [(0,2),(0,3),(1,2),(1,3)]
                    # c2_idx 2 -> C10 block, positions [(2,0),(2,1),(3,0),(3,1)]
                    # c2_idx 3 -> C11 block, positions [(2,2),(2,3),(3,2),(3,3)]
                    
                    c_block_positions = {
                        0: [(0,0), (0,1), (1,0), (1,1)],
                        1: [(0,2), (0,3), (1,2), (1,3)],
                        2: [(2,0), (2,1), (3,0), (3,1)],
                        3: [(2,2), (2,3), (3,2), (3,3)],
                    }
                    
                    for (row, col) in c_block_positions[c2_idx]:
                        # W index: k*n + i where k is col, i is row
                        # W[k*n + i, col]
                        w_vec[col * n + row] += c2_coeff * c_contribs[mi].get(c2_idx, 0)
                
                # Wait, I'm confusing myself. Let me reconsider.
                # c2_contribs gives the contribution of this 2x2 Strassen result to the 4x4 C.
                # But c2_contribs is for the 2x2 result of the top-level multiplication.
                # The top-level multiplication M_mi produces a 2x2 matrix.
                # c2_contribs[0] = contribution to C00 (top-left 2x2)
                # c2_contribs[1] = contribution to C01 (top-right 2x2)
                # c2_contribs[2] = contribution to C10 (bottom-left 2x2)
                # c2_contribs[3] = contribution to C11 (bottom-right 2x2)
                
                # But c_contribs[mi] gives the contribution of M_mi to the C blocks.
                # I think I'm double-counting. Let me reconsider.
                
                # Actually, c2_contribs is the contribution within the 2x2 Strassen decomposition.
                # And c_contribs[mi] is the contribution of the entire M_mi to the 4x4 C.
                # These are different things.
                
                # Let me reconsider the structure:
                # Top-level: M_mi = A'_mi * B'_mi (2x2 matrix product)
                # The result M_mi is a 2x2 matrix.
                # c_contribs[mi] tells us how M_mi contributes to C00, C01, C10, C11.
                # e.g., c_contribs[0] = {0: 1, 3: 1} means M1 contributes +1 to C00 and +1 to C11.
                
                # Now, M_mi itself is computed via 2x2 Strassen, giving 7 scalar products.
                # Each scalar product contributes to the 2x2 result M_mi.
                # c2_contribs tells us which elements of the 2x2 result M_mi each scalar product contributes to.
                
                # So the total contribution of a scalar product to C is:
                # For each element (r,s) of M_mi that the scalar product contributes to (with coeff c2_contribs[r*2+s]),
                # and for each C block that M_mi contributes to (with coeff c_contribs[mi][block]),
                # the scalar product contributes c2_coeff * c_contrib to C[block_row+r, block_col+s].
                
                # I think I need to multiply the contributions.
                
                # Let me redo the W vector construction.
                w_vec = [0] * (4 * n)  # 16
                
                for c2_idx, c2_coeff in c2_contribs.items():
                    if c2_coeff == 0:
                        continue
                    # c2_idx: 0=C00', 1=C01', 2=C10', 3=C11'
                    # This is an element of the 2x2 result M_mi
                    # c_contribs[mi] gives how M_mi contributes to C blocks
                    # But c_contribs[mi] is a dict from C block index to coefficient
                    # And c2_idx is also a C block index? No...
                    
                    # Wait, I think I've been confused. Let me re-examine.
                    # The 2x2 Strassen decomposition computes a 2x2 matrix M_mi.
                    # The 7 scalar products each contribute to specific elements of M_mi.
                    # c2_contribs maps: scalar product -> {element_of_M_mi: coefficient}
                    # where element_of_M_mi is 0,1,2,3 for (0,0),(0,1),(1,0),(1,1)
                    
                    # Then, the 2x2 matrix M_mi is used in the 4x4 C reconstruction:
                    # C00 = M1 + M4 - M5 + M7  (each M is 2x2, added element-wise)
                    # So M_mi[r,s] contributes to C[block_row+mi_block_r, block_col+mi_block_s]
                    # where mi_block_r, mi_block_s depend on which C block.
                    
                    # Actually, c_contribs[mi] should tell us which C block M_mi goes to.
                    # But M_mi is a 2x2 matrix, and it contributes to multiple C blocks?
                    # No, each M_mi contributes to specific C blocks.
                    
                    # Looking at the C reconstruction:
                    # C00 = M1 + M4 - M5 + M7  -> M1, M4, M5, M7 all contribute to C00
                    # C01 = M3 + M5  -> M3, M5 contribute to C01
                    # C10 = M2 + M4  -> M2, M4 contribute to C10
                    # C11 = M1 - M2 + M3 + M6  -> M1, M2, M3, M6 contribute to C11
                    
                    # So M1 contributes to both C00 and C11.
                    # c_contribs[0] = {0: 1, 3: 1} means M1[0,0] contributes to C00[0,0] and C11[0,0], etc.
                    
                    # So for a scalar product that contributes to M_mi[r,s] with coefficient c2_coeff,
                    # and M_mi contributes to C block b with coefficient c_contrib,
                    # the scalar product contributes c2_coeff * c_contrib to C[b_row+r, b_col+s].
                    
                    # But c_contribs[mi] is a dict from C block index to coefficient.
                    # And c2_idx is the element index within M_mi (0,1,2,3 for (0,0),(0,1),(1,0),(1,1)).
                    
                    # So I need to iterate over c_contribs[mi] items, and for each C block,
                    # add the contribution to the appropriate C positions.
                    
                    # Let me redo:
                    w_vec = [0] * (4 * n)
                    
                    for c_block_idx, c_block_coeff in c_contribs[mi].items():
                        if c_block_coeff == 0:
                            continue
                        # c_block_idx: 0=C00, 1=C01, 2=C10, 3=C11
                        # c2_idx: element within M_mi (0,1,2,3)
                        # The C position is:
                        # C00: rows 0-1, cols 0-1
                        # C01: rows 0-1, cols 2-3
                        # C10: rows 2-3, cols 0-1
                        # C11: rows 2-3, cols 2-3
                        
                        c_block_row_offset = [0, 0, 2, 2][c_block_idx]
                        c_block_col_offset = [0, 2, 0, 2][c_block_idx]
                        
                        # c2_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                        r = c2_idx // 2
                        s = c2_idx % 2
                        
                        row = c_block_row_offset + r
                        col = c_block_col_offset + s
                        
                        w_vec[col * n + row] += c2_coeff * c_block_coeff
                
                # Now add to U, V, W
                for idx in range(n * m):
                    U[idx][col] = u_vec[idx]
                for idx in range(m * 4):
                    V[idx + B_col_offset * 0][col] = v_vec[idx]  # B_col_offset already in v_vec
                # Wait, v_vec already has the offset. Let me fix.
                
                # Actually, v_vec is indexed by [row * 4 + col + B_col_offset]
                # But V has shape (m*p, rank) = (36, rank)
                # For B1 (first 4x4), B_col_offset = 0, so V indices 0-15
                # For B2 (second 4x4), B_col_offset = 4, so V indices 4-19
                # This is correct.
                
                for idx in range(m * p):
                    V[idx][col] = v_vec[idx] if idx < len(v_vec) else 0
                for idx in range(p * n):
                    W[idx][col] = w_vec[idx]
                
                col += 1
        
        return col
    
    # Add Strassen for A * B1 (B1 is columns 0-3 of B)
    col = add_strassen_4x4x4(col, 0)
    
    # Add Strassen for A * B2 (B2 is columns 4-7 of B)
    col = add_strassen_4x4x4(col, 4)
    
    # Add schoolbook for A * B3 (B3 is column 8 of B, 4x1)
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 8][col] = 1  # B3 is at column 8
            W[8 * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
