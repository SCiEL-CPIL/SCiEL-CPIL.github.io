def construct():
    n, m, p = 4, 4, 8
    rank = 98
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    col = 0
    
    def add_strassen_4x4(B_col_offset):
        """Add Strassen decomposition for 4x4 x 4x4 where B is columns B_col_offset to B_col_offset+3."""
        nonlocal col
        # A is full 4x4, B is 4x4 starting at column B_col_offset
        
        # Define the 7 Strassen block products as (A_factor, B_factor) pairs
        # Each factor is a 2x2 matrix of linear forms
        
        # M1 = (A11+A22)(B11+B22)
        # A11+A22: row i in 0-1, col j in 0-1: A[i,j] + A[i+2,j]
        # B11+B22: row j in 0-1, col k in 0-1: B[j, B_col_offset+k] + B[j+2, B_col_offset+k]
        
        strassen_products = []
        
        # M1: (A11+A22)(B11+B22)
        A_M1 = {}
        for i in range(2):
            for j in range(2):
                A_M1[(i,j)] = [(i*4+j, 1), ((i+2)*4+j, 1)]
        B_M1 = {}
        for j in range(2):
            for k in range(2):
                B_M1[(j,k)] = [(j*8+B_col_offset+k, 1), ((j+2)*8+B_col_offset+k, 1)]
        strassen_products.append((A_M1, B_M1))
        
        # M2: (A21+A22)(B11)
        A_M2 = {}
        for i in range(2):
            for j in range(2):
                A_M2[(i,j)] = [((i+2)*4+j, 1)]  # A21: rows 2-3, cols 0-1; A22: rows 2-3, cols 2-3
                # Wait, A21 is A[2:4, 0:2], A22 is A[2:4, 2:4]
                # A21+A22: A[2:4, 0:4]
                A_M2[(i,j)] = [((i+2)*4+j, 1)]
        # Actually A21+A22 means for each position (i,j) in 0-1, 0-1: A[i+2, j] + A[i+2, j+2]
        A_M2 = {}
        for i in range(2):
            for j in range(2):
                A_M2[(i,j)] = [((i+2)*4+j, 1), ((i+2)*4+(j+2), 1)]
        B_M2 = {}
        for j in range(2):
            for k in range(2):
                B_M2[(j,k)] = [(j*8+B_col_offset+k, 1)]  # B11: rows 0-1, cols 0-1
        strassen_products.append((A_M2, B_M2))
        
        # M3: (A11)(B12-B22)
        A_M3 = {}
        for i in range(2):
            for j in range(2):
                A_M3[(i,j)] = [(i*4+j, 1)]
        B_M3 = {}
        for j in range(2):
            for k in range(2):
                B_M3[(j,k)] = [(j*8+B_col_offset+2+k, 1), ((j+2)*8+B_col_offset+2+k, -1)]
        strassen_products.append((A_M3, B_M3))
        
        # M4: (A22)(B21-B11)
        A_M4 = {}
        for i in range(2):
            for j in range(2):
                A_M4[(i,j)] = [((i+2)*4+(j+2), 1)]
        B_M4 = {}
        for j in range(2):
            for k in range(2):
                B_M4[(j,k)] = [((j+2)*8+B_col_offset+k, 1), (j*8+B_col_offset+k, -1)]
        strassen_products.append((A_M4, B_M4))
        
        # M5: (A11+A12)(B22)
        A_M5 = {}
        for i in range(2):
            for j in range(2):
                A_M5[(i,j)] = [(i*4+j, 1), (i*4+(j+2), 1)]
        B_M5 = {}
        for j in range(2):
            for k in range(2):
                B_M5[(j,k)] = [((j+2)*8+B_col_offset+2+k, 1)]
        strassen_products.append((A_M5, B_M5))
        
        # M6: (A21-A11)(B11+B12)
        A_M6 = {}
        for i in range(2):
            for j in range(2):
                A_M6[(i,j)] = [((i+2)*4+j, 1), (i*4+j, -1)]
        B_M6 = {}
        for j in range(2):
            for k in range(2):
                B_M6[(j,k)] = [(j*8+B_col_offset+k, 1), (j*8+B_col_offset+2+k, 1)]
        strassen_products.append((A_M6, B_M6))
        
        # M7: (A12-A22)(B21+B22)
        A_M7 = {}
        for i in range(2):
            for j in range(2):
                A_M7[(i,j)] = [(i*4+(j+2), 1), ((i+2)*4+(j+2), -1)]
        B_M7 = {}
        for j in range(2):
            for k in range(2):
                B_M7[(j,k)] = [((j+2)*8+B_col_offset+k, 1), ((j+2)*8+B_col_offset+2+k, 1)]
        strassen_products.append((A_M7, B_M7))
        
        # Now for each of the 7 block products, apply Strassen 2x2 x 2x2
        # and track how the results combine into C
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Each Mi is a 2x2 matrix. For Mi = Xi * Yi, the Strassen gives:
        # Mi[0,0] = S1 + S4 - S5 + S7
        # Mi[0,1] = S3 + S5
        # Mi[1,0] = S2 + S4
        # Mi[1,1] = S1 - S2 + S3 + S6
        
        # Where S1..S7 are the 7 scalar products.
        
        # For each block product index bi (0-6), and each Strassen scalar product si (0-6),
        # we have a rank-1 term.
        
        # The C contribution: for block product bi, the 2x2 result contributes to C blocks.
        # C11 gets contribution from M1[0,0], M4[0,0], -M5[0,0], M7[0,0]
        # etc.
        
        # Let me define for each block product bi, which C entries it contributes to and with what coefficient
        # C11 = M1 + M4 - M5 + M7: C11[0,0] += M1[0,0], C11[0,1] += M1[0,1], etc.
        # Actually C11 is a 2x2 block, so C11[i,j] = M1[i,j] + M4[i,j] - M5[i,j] + M7[i,j]
        
        c_block_contrib = {
            0: [(0, 1), (3, 1), (4, -1), (6, 1)],   # M1, M4, -M5, M7 for C11 (block 0)
            1: [(2, 1), (4, 1)],                       # M3, M5 for C12 (block 1)
            2: [(1, 1), (3, 1)],                       # M2, M4 for C21 (block 2)
            3: [(0, 1), (1, -1), (2, 1), (5, 1)],     # M1, -M2, M3, M6 for C22 (block 3)
        }
        
        # Strassen 2x2 output coefficients:
        # S1 -> [0,0] and [1,1]
        # S2 -> [1,0] and [1,1]
        # S3 -> [0,1] and [1,1]
        # S4 -> [0,0] and [1,0]
        # S5 -> [0,0] and [0,1]
        # S6 -> [1,1]
        # S7 -> [0,0]
        
        # More precisely:
        # Z[0,0] = S1 + S4 - S5 + S7
        # Z[0,1] = S3 + S5
        # Z[1,0] = S2 + S4
        # Z[1,1] = S1 - S2 + S3 + S6
        
        strassen_out_coeffs = {
            0: [(0, 1), (3, 1), (4, -1), (6, 1)],  # S1->Z[0,0], S4->Z[0,0], S5->Z[0,0], S7->Z[0,0]
            1: [(3, 1), (4, 1)],                     # S3->Z[0,1], S5->Z[0,1]
            2: [(1, 1), (3, 1)],                     # S2->Z[1,0], S4->Z[1,0]
            3: [(0, 1), (1, -1), (2, 1), (5, 1)],   # S1->Z[1,1], S2->Z[1,1], S3->Z[1,1], S6->Z[1,1]
        }
        
        for bi in range(7):
            A_bi, B_bi = strassen_products[bi]
            # Apply Strassen 2x2 to A_bi * B_bi
            # A_bi is a dict (i,j) -> list of (A_index, coeff)
            # B_bi is a dict (j,k) -> list of (B_index, coeff)
            
            # Extract the 4 entries of A_bi and B_bi as linear forms
            # A_bi[i][j] for i,j in 0-1
            A_entries = {}
            for i in range(2):
                for j in range(2):
                    A_entries[(i,j)] = A_bi[(i,j)]
            B_entries = {}
            for j in range(2):
                for k in range(2):
                    B_entries[(j,k)] = B_bi[(j,k)]
            
            # Strassen 2x2:
            # x11 = A[0,0], x12 = A[0,1], x21 = A[1,0], x22 = A[1,1]
            # y11 = B[0,0], y12 = B[0,1], y21 = B[1,0], y22 = B[1,1]
            
            x11 = A_entries[(0,0)]
            x12 = A_entries[(0,1)]
            x21 = A_entries[(1,0)]
            x22 = A_entries[(1,1)]
            y11 = B_entries[(0,0)]
            y12 = B_entries[(0,1)]
            y21 = B_entries[(1,0)]
            y22 = B_entries[(1,1)]
            
            # S1 = (x11+x22)(y11+y22)
            s1_A = x11 + x22
            s1_B = y11 + y22
            # S2 = (x21+x22)(y11)
            s2_A = x21 + x22
            s2_B = y11
            # S3 = (x11)(y12-y22)
            s3_A = x11
            s3_B = y12 + [(-idx, -c) for idx, c in y22]
            # S4 = (x22)(y21-y11)
            s4_A = x22
            s4_B = y21 + [(-idx, -c) for idx, c in y11]
            # S5 = (x11+x12)(y22)
            s5_A = x11 + x12
            s5_B = y22
            # S6 = (x21-x11)(y11+y12)
            s6_A = x21 + [(-idx, -c) for idx, c in x11]
            s6_B = y11 + y12
            # S7 = (x12-x22)(y21+y22)
            s7_A = x12 + [(-idx, -c) for idx, c in x22]
            s7_B = y21 + y22
            
            s_A = [s1_A, s2_A, s3_A, s4_A, s5_A, s6_A, s7_A]
            s_B = [s1_B, s2_B, s3_B, s4_B, s5_B, s6_B, s7_B]
            
            for si in range(7):
                # Add rank-1 term for this scalar multiplication
                for (a_idx, a_coeff) in s_A[si]:
                    U[a_idx][col] += a_coeff
                for (b_idx, b_coeff) in s_B[si]:
                    V[b_idx][col] += b_coeff
                
                # Now add C contributions
                # This scalar product contributes to Z[zi,zj] with coefficient from strassen_out_coeffs
                # And Z is block bi of the 4x4 result
                # Block bi corresponds to C block: bi=0->C11, bi=1->C12, bi=2->C21, bi=3->C22
                # C block (ci,cj) is at C[ci*2:ci*2+2, cj*2:cj*2+2]
                # C index = (cj*4+ci)*8 + ... wait, C is 4x8, indexed as C[i,k] where i=0-3, k=0-7
                # C block (ci,cj) has C[ci*2+di, cj*2+dj] for di,dj in 0-1
                
                # For block product bi, it contributes to C blocks according to c_block_contrib[bi]
                # Wait, I defined c_block_contrib differently. Let me redo.
                
                # The 4x4 result C has blocks:
                # C[0:2, 0:2] = C11, C[0:2, 2:4] = C12, C[2:4, 0:2] = C21, C[2:4, 2:4] = C22
                # C11 = M1 + M4 - M5 + M7
                # C12 = M3 + M5
                # C21 = M2 + M4
                # C22 = M1 - M2 + M3 + M6
                
                # So block product bi contributes to C blocks:
                block_to_c = {
                    0: [(0, 1), (3, 1), (4, -1), (6, 1)],  # M1->C11, M4->C11, M5->C11(-1), M7->C11
                    1: [(2, 1), (4, 1)],                      # M3->C12, M5->C12
                    2: [(1, 1), (3, 1)],                      # M2->C21, M4->C21
                    3: [(0, 1), (1, -1), (2, 1), (5, 1)],    # M1->C22, M2->C22(-1), M3->C22, M6->C22
                }
                
                # Wait, this is wrong. block_to_c should map from block product index to which C blocks it affects.
                # M1 (bi=0) affects C11 and C22
                # M2 (bi=1) affects C21 and C22
                # M3 (bi=2) affects C12 and C22
                # M4 (bi=3) affects C11 and C21
                # M5 (bi=4) affects C11 and C12
                # M6 (bi=5) affects C22
                # M7 (bi=6) affects C11
                
                m_to_c = {
                    0: [(0, 1), (3, 1)],    # M1 -> C11(+1), C22(+1)
                    1: [(2, 1), (3, -1)],   # M2 -> C21(+1), C22(-1)
                    2: [(1, 1), (3, 1)],    # M3 -> C12(+1), C22(+1)
                    3: [(0, 1), (2, 1)],    # M4 -> C11(+1), C21(+1)
                    4: [(0, -1), (1, 1)],   # M5 -> C11(-1), C12(+1)
                    5: [(3, 1)],            # M6 -> C22(+1)
                    6: [(0, 1)],            # M7 -> C11(+1)
                }
                
                for (c_block, c_coeff) in m_to_c[bi]:
                    # c_block: 0=C11, 1=C12, 2=C21, 3=C22
                    # The 2x2 result Z from this block product contributes to C block c_block
                    # Z[di,dj] contributes to C[c_block_row*2+di, c_block_col*2+dj]
                    c_block_rows = {0: 0, 1: 0, 2: 2, 3: 2}
                    c_block_cols = {0: 0, 1: 2, 2: 0, 3: 2}
                    cr = c_block_rows[c_block]
                    cc = c_block_cols[c_block]
                    
                    for (zi, zj, z_coeff) in [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]:
                        # Z[zi,zj] has coefficient from strassen_out_coeffs
                        # strassen_out_coeffs[zi*2+zj] gives list of (si, coeff)
                        total_coeff = 0
                        for (si2, coeff2) in strassen_out_coeffs[zi*2+zj]:
                            if si2 == si:
                                total_coeff += coeff2
                        if total_coeff != 0:
                            c_idx = (cc + zj)*4 + (cr + zi)  # W index: k*n+i = k*4+i
                            W[c_idx][col] += c_coeff * total_coeff
                
                col += 1
        
        # col should have increased by 49 (7 block products * 7 scalar mults)
    
    add_strassen_4x4(0)  # Left half: B columns 0-3
    add_strassen_4x4(4)  # Right half: B columns 4-7
    
    return {"U": U, "V": V, "W": W}
