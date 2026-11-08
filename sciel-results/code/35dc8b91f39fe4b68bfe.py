"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    # Strategy:
    # Split B into B1 (4x4) and B2 (4x5)
    # C1 (4x4) = A * B1 : use recursive Strassen, rank 49
    # C2 (4x5) = A * B2 : schoolbook, rank 20
    # Total rank: 49 + 20 = 69
    
    rank = 69
    U = [[0] * rank for _ in range(n * m)]  # 16 x 69
    V = [[0] * rank for _ in range(m * p)]  # 36 x 69
    W = [[0] * rank for _ in range(p * n)]  # 36 x 69
    
    col = 0
    
    # Part 1: C[:, 0:4] = A * B[:, 0:4] using recursive Strassen (rank 49)
    # Recursive Strassen: split 4x4 into 2x2 blocks of 2x2 submatrices
    # Apply Strassen at block level (7 block multiplications)
    # Each block multiplication is 2x2 * 2x2, done with Strassen (rank 7)
    
    # Define Strassen 2x2x2 decomposition
    # For 2x2: A=[[a0,a1],[a2,a3]], B=[[b0,b1],[b2,b3]], C=A*B
    # p1=(a0+a3)(b0+b3), p2=(a2+a3)b0, p3=a0(b1-b3), p4=a3(b2-b0)
    # p5=(a0+a1)b3, p6=(a2-a0)(b0+b1), p7=(a1-a3)(b2+b3)
    # c00=p1+p4-p5+p7, c01=p3+p5, c10=p2+p4, c11=p1-p2+p3+p6
    
    # For 4x4x4 recursive Strassen:
    # A blocks: A00(0-1,0-1), A01(0-1,2-3), A10(2-3,0-1), A11(2-3,2-3)
    # B blocks: B00(0-1,0-1), B01(0-1,2-3), B10(2-3,0-1), B11(2-3,2-3)
    # C blocks: C00(0-1,0-1), C01(0-1,2-3), C10(2-3,0-1), C11(2-3,2-3)
    
    # Block-level Strassen:
    # M1=(A00+A11)*(B00+B11) -> contributes to C00, C11
    # M2=(A10+A11)*B00 -> contributes to C10, C11
    # M3=A00*(B01-B11) -> contributes to C01, C11
    # M4=A11*(B10-B00) -> contributes to C00, C10
    # M5=(A00+A01)*B11 -> contributes to C01, C00
    # M6=(A10-A00)*(B00+B01) -> contributes to C11
    # M7=(A01-A11)*(B10+B11) -> contributes to C00, C01
    
    # Each Mi is a 2x2*2x2 product, decomposed with Strassen (7 terms each)
    # Total: 7 * 7 = 49 terms
    
    # Helper: add 7 Strassen terms for 2x2*2x2 at given positions
    def add_strassen_2x2(ar, ac, br, bc, cr, cc, offset, u_coeff, v_coeff, w_coeff):
        """Add 7 Strassen rank-1 terms for 2x2 block product.
        ar,ac: top-left of A block (2x2)
        br,bc: top-left of B block (2x2)
        cr,cc: top-left of C block (2x2)
        offset: starting column
        u_coeff, v_coeff, w_coeff: scalar multipliers for U, V, W
        Returns offset+7
        """
        for t in range(7):
            c = offset + t
            if t == 0:
                # p1=(a0+a3)(b0+b3)
                U[ar*m+ac][c] = u_coeff
                U[(ar+1)*m+(ac+1)][c] = u_coeff
                V[br*p+bc][c] = v_coeff
                V[(br+1)*p+(bc+1)][c] = v_coeff
                W[cr*n+cc][c] = w_coeff
                W[(cr+1)*n+(cc+1)][c] = w_coeff
            elif t == 1:
                # p2=(a2+a3)b0
                U[(ar+1)*m+ac][c] = u_coeff
                U[(ar+1)*m+(ac+1)][c] = u_coeff
                V[br*p+bc][c] = v_coeff
                W[(cr+1)*n+cc][c] = w_coeff
                W[(cr+1)*n+(cc+1)][c] = -w_coeff
            elif t == 2:
                # p3=a0(b1-b3)
                U[ar*m+ac][c] = u_coeff
                V[br*p+(bc+1)][c] = v_coeff
                V[(br+1)*p+(bc+1)][c] = -v_coeff
                W[cr*n+(cc+1)][c] = w_coeff
                W[(cr+1)*n+(cc+1)][c] = w_coeff
            elif t == 3:
                # p4=a3(b2-b0)
                U[(ar+1)*m+(ac+1)][c] = u_coeff
                V[(br+1)*p+bc][c] = v_coeff
                V[br*p+bc][c] = -v_coeff
                W[cr*n+cc][c] = w_coeff
                W[(cr+1)*n+cc][c] = w_coeff
            elif t == 4:
                # p5=(a0+a1)b3
                U[ar*m+ac][c] = u_coeff
                U[ar*m+(ac+1)][c] = u_coeff
                V[(br+1)*p+(bc+1)][c] = v_coeff
                W[cr*n+(cc+1)][c] = w_coeff
                W[cr*n+cc][c] = -w_coeff
            elif t == 5:
                # p6=(a2-a0)(b0+b1)
                U[(ar+1)*m+ac][c] = u_coeff
                U[ar*m+ac][c] = -u_coeff
                V[br*p+bc][c] = v_coeff
                V[br*p+(bc+1)][c] = v_coeff
                W[(cr+1)*n+(cc+1)][c] = w_coeff
            elif t == 6:
                # p7=(a1-a3)(b2+b3)
                U[ar*m+(ac+1)][c] = u_coeff
                U[(ar+1)*m+(ac+1)][c] = -u_coeff
                V[(br+1)*p+bc][c] = v_coeff
                V[(br+1)*p+(bc+1)][c] = v_coeff
                W[cr*n+cc][c] = w_coeff
                W[cr*n+(cc+1)][c] = w_coeff
        return offset + 7
    
    # Now build the 49 terms for 4x4x4
    # M1=(A00+A11)*(B00+B11) -> C00 += M1, C11 += M1
    # We need to express (A00+A11) as a 2x2 matrix and (B00+B11) as a 2x2 matrix
    # Then apply Strassen to get 7 terms
    
    # For M1: U part = A00+A11, V part = B00+B11, W part = C00 and C11
    # A00 is at (0,0), A11 is at (2,2)
    # B00 is at (0,0), B11 is at (2,2)
    # C00 is at (0,0), C11 is at (2,2)
    
    # For the U matrix of M1's Strassen terms:
    # The "A" matrix for Strassen is A00+A11, so:
    # (A00+A11)[0,0] = A[0,0]+A[2,2]
    # (A00+A11)[0,1] = A[0,1]+A[2,3]
    # (A00+A11)[1,0] = A[1,0]+A[3,2]
    # (A00+A11)[1,1] = A[1,1]+A[3,3]
    
    # Similarly for V and W.
    
    # I'll handle each of the 7 block-level Strassen terms by specifying
    # the linear forms explicitly.
    
    # Let me define a more flexible helper that takes explicit U, V, W coefficient dicts
    
    def add_term(u_coeffs, v_coeffs, w_coeffs, offset):
        """Add one rank-1 term with given coefficient dictionaries."""
        c = offset
        for (i, j), val in u_coeffs.items():
            U[i*m+j][c] += val
        for (j, k), val in v_coeffs.items():
            V[j*p+k][c] += val
        for (k, i), val in w_coeffs.items():
            W[k*n+i][c] += val
        return c + 1
    
    # M1 = (A00+A11)*(B00+B11)
    # U linear forms (for the 2x2 "A" = A00+A11):
    #   a00 = A[0,0]+A[2,2], a01 = A[0,1]+A[2,3]
    #   a10 = A[1,0]+A[3,2], a11 = A[1,1]+A[3,3]
    # V linear forms (for the 2x2 "B" = B00+B11):
    #   b00 = B[0,0]+B[2,2], b01 = B[0,1]+B[2,3]
    #   b10 = B[1,0]+B[3,2], b11 = B[1,1]+B[3,3]
    # W: C00 and C11 get contributions
    
    # Strassen p1=(a00+a11)(b00+b11):
    # U: A[0,0]+A[2,2]+A[1,1]+A[3,3]
    # V: B[0,0]+B[2,2]+B[1,1]+B[3,3]
    # W: C[0,0]+=1, C[3,3]+=1 (from C11 contribution) and C[0,0]+=1, C[1,1]+=1 (from C00)
    # Wait, C00 is at positions (0,0),(0,1),(1,0),(1,1) and C11 is at (2,2),(2,3),(3,2),(3,3)
    
    # Let me be very explicit. I'll enumerate all 49 terms.
    
    # Define the 7 Strassen products for a generic 2x2 with entries indexed [0..3]:
    # A=[a0,a1,a2,a3] = [[a0,a1],[a2,a3]]
    # B=[b0,b1,b2,b3] = [[b0,b1],[b2,b3]]
    # p0=(a0+a3)(b0+b3)
    # p1=(a2+a3)b0
    # p2=a0(b1-b3)
    # p3=a3(b2-b0)
    # p4=(a0+a1)b3
    # p5=(a2-a0)(b0+b1)
    # p6=(a1-a3)(b2+b3)
    # c0=p0+p3-p4+p6
    # c1=p2+p4
    # c2=p1+p3
    # c3=p0-p1+p2+p5
    
    # For each block-level Strassen multiplication Mi, I define:
    # - The 4 A-entries as linear combinations of A matrix entries
    # - The 4 B-entries as linear combinations of B matrix entries
    # - The C output positions and coefficients
    
    # M1=(A00+A11)*(B00+B11):
    # A-entries: a0=A[0,0]+A[2,2], a1=A[0,1]+A[2,3], a2=A[1,0]+A[3,2], a3=A[1,1]+A[3,3]
    # B-entries: b0=B[0,0]+B[2,2], b1=B[0,1]+B[2,3], b2=B[1,0]+B[3,2], b3=B[1,1]+B[3,3]
    # C contributions: C00 gets (c0,c1,c2,c3) at (0,0),(0,1),(1,0),(1,1)
    #                  C11 gets (c0,c1,c2,c3) at (2,2),(2,3),(3,2),(3,3)
    
    # M2=(A10+A11)*B00:
    # A-entries: a0=A[2,0]+A[0,0], a1=A[2,1]+A[0,1], a2=A[3,0]+A[1,0], a3=A[3,1]+A[1,1]
    # B-entries: b0=B[0,0], b1=B[0,1], b2=B[1,0], b3=B[1,1]
    # C contributions: C10 gets (c0,c1,c2,c3) at (2,0),(2,1),(3,0),(3,1)
    #                  C11 gets -(c0,c1,c2,c3) at (2,2),(2,3),(3,2),(3,3)
    
    # M3=A00*(B01-B11):
    # A-entries: a0=A[0,0], a1=A[0,1], a2=A[1,0], a3=A[1,1]
    # B-entries: b0=B[0,2]-B[2,2], b1=B[0,3]-B[2,3], b2=B[1,2]-B[3,2], b3=B[1,3]-B[3,3]
    # C contributions: C01 gets (c0,c1,c2,c3) at (0,2),(0,3),(1,2),(1,3)
    #                  C11 gets (c0,c1,c2,c3) at (2,2),(2,3),(3,2),(3,3)
    
    # M4=A11*(B10-B00):
    # A-entries: a0=A[2,2], a1=A[2,3], a2=A[3,2], a3=A[3,3]
    # B-entries: b0=B[2,0]-B[0,0], b1=B[2,1]-B[0,1], b2=B[3,0]-B[1,0], b3=B[3,1]-B[1,1]
    # C contributions: C00 gets (c0,c1,c2,c3) at (0,0),(0,1),(1,0),(1,1)
    #                  C10 gets (c0,c1,c2,c3) at (2,0),(2,1),(3,0),(3,1)
    
    # M5=(A00+A01)*B11:
    # A-entries: a0=A[0,0]+A[0,2], a1=A[0,1]+A[0,3], a2=A[1,0]+A[1,2], a3=A[1,1]+A[1,3]
    # B-entries: b0=B[2,2], b1=B[2,3], b2=B[3,2], b3=B[3,3]
    # C contributions: C01 gets (c0,c1,c2,c3) at (0,2),(0,3),(1,2),(1,3)
    #                  C00 gets -(c0,c1,c2,c3) at (0,0),(0,1),(1,0),(1,1)
    
    # M6=(A10-A00)*(B00+B01):
    # A-entries: a0=A[2,0]-A[0,0], a1=A[2,1]-A[0,1], a2=A[3,0]-A[1,0], a3=A[3,1]-A[1,1]
    # B-entries: b0=B[0,0]+B[0,2], b1=B[0,1]+B[0,3], b2=B[1,0]+B[1,2], b3=B[1,1]+B[1,3]
    # C contributions: C11 gets (c0,c1,c2,c3) at (2,2),(2,3),(3,2),(3,3)
    
    # M7=(A01-A11)*(B10+B11):
    # A-entries: a0=A[0,2]-A[2,2], a1=A[0,3]-A[2,3], a2=A[1,2]-A[3,2], a3=A[1,3]-A[3,3]
    # B-entries: b0=B[2,0]+B[2,2], b1=B[2,1]+B[2,3], b2=B[3,0]+B[3,2], b3=B[3,1]+B[3,3]
    # C contributions: C00 gets (c0,c1,c2,c3) at (0,0),(0,1),(1,0),(1,1)
    #                  C01 gets (c0,c1,c2,c3) at (0,2),(0,3),(1,2),(1,3)
    
    # Now I'll write a function to add 7 Strassen terms given the A, B linear forms and C output spec
    
    def add_block_strassen(a_forms, b_forms, c_outputs, offset):
        """
        a_forms: dict mapping (row,col) in 2x2 to list of (A_row, A_col, coeff)
        b_forms: dict mapping (row,col) in 2x2 to list of (B_row, B_col, coeff)  
        c_outputs: list of (c_row_offset, c_col_offset, sign) for the 4 C entries
                   where c0->(cr0,cc0,sign0), c1->(cr1,cc1,sign1), etc.
                   Actually let me use: list of 4 tuples (c_row, c_col, coeff)
                   for c0, c1, c2, c3 respectively
        """
        # a_forms: {0: [(ar,ac,coef),...], 1: [...], 2: [...], 3: [...]}
        # b_forms: {0: [(br,bc,coef),...], 1: [...], 2: [...], 3: [...]}
        # c_outputs: [(cr,cc,coef), (cr,cc,coef), (cr,cc,coef), (cr,cc,coef)] for c0,c1,c2,c3
        
        # The 7 Strassen products:
        # p0=(a0+a3)(b0+b3) -> c0+=1, c3+=1
        # p1=(a2+a3)b0 -> c2+=1, c3+=-1
        # p2=a0(b1-b3) -> c1+=1, c3+=1
        # p3=a3(b2-b0) -> c0+=1, c2+=1
        # p4=(a0+a1)b3 -> c1+=1, c0+=-1
        # p5=(a2-a0)(b0+b1) -> c3+=1
        # p6=(a1-a3)(b2+b3) -> c0+=1, c1+=1
        
        strassen_defs = [
            # (a_indices, a_coeffs, b_indices, b_coeffs, c_contribs)
            # c_contribs: list of (c_idx, coeff) where c_idx is 0,1,2,3
            ({0:1, 3:1}, {0:1, 3:1}, [(0,1),(3,1)]),  # p0
            ({2:1, 3:1}, {0:1}, [(2,1),(3,-1)]),       # p1
            ({0:1}, {1:1, 3:-1}, [(1,1),(3,1)]),       # p2
            ({3:1}, {2:1, 0:-1}, [(0,1),(2,1)]),       # p3
            ({0:1, 1:1}, {3:1}, [(1,1),(0,-1)]),       # p4
            ({2:1, 0:-1}, {0:1, 1:1}, [(3,1)]),        # p5
            ({1:1, 3:-1}, {2:1, 3:1}, [(0,1),(1,1)]),  # p6
        ]
        
        for a_dict, b_dict, c_contribs in strassen_defs:
            c = offset
            # Build U coefficients
            u_coeffs = {}
            for a_idx, a_coef in a_dict.items():
                for (ar, ac, form_coef) in a_forms[a_idx]:
                    key = (ar, ac)
                    u_coeffs[key] = u_coeffs.get(key, 0) + a_coef * form_coef
            for (ar, ac), val in u_coeffs.items():
                U[ar*m+ac][c] += val
            
            # Build V coefficients
            v_coeffs = {}
            for b_idx, b_coef in b_dict.items():
                for (br, bc, form_coef) in b_forms[b_idx]:
                    key = (br, bc)
                    v_coeffs[key] = v_coeffs.get(key, 0) + b_coef * form_coef
            for (br, bc), val in v_coeffs.items():
                V[br*p+bc][c] += val
            
            # Build W coefficients
            for c_idx, c_coef in c_contribs:
                cr, cc, out_coef = c_outputs[c_idx]
                W[cr*n+cc][c] += c_coef * out_coef
            
            offset += 1
        
        return offset
    
    # Define the 7 block-level Strassen multiplications
    
    # M1=(A00+A11)*(B00+B11)
    a1 = {0: [(0,0,1),(2,2,1)], 1: [(0,1,1),(2,3,1)], 2: [(1,0,1),(3,2,1)], 3: [(1,1,1),(3,3,1)]}
    b1 = {0: [(0,0,1),(2,2,1)], 1: [(0,1,1),(2,3,1)], 2: [(1,0,1),(3,2,1)], 3: [(1,1,1),(3,3,1)]}
    c1 = [(0,0,1),(0,1,1),(1,0,1),(1,1,1)]  # C00
    # But M1 also contributes to C11 with same coefficients
    # I need to handle multiple C outputs. Let me modify the approach.
    
    # Actually, let me just handle each C output separately by adding terms.
    # For M1, C00 gets the Strassen result AND C11 gets the Strassen result.
    # So I'll add 7 terms for C00 contribution and 7 terms for C11 contribution? No,
    # that would be 14 terms. Instead, each Strassen term contributes to BOTH C00 and C11.
    
    # Let me modify: c_outputs should be a list of 4 entries, each being a list of (cr,cc,coef)
    
    def add_block_strassen_v2(a_forms, b_forms, c_outputs, offset):
        """
        c_outputs: list of 4 lists, each list contains (cr, cc, coef) for that c entry
        """
        strassen_defs = [
            ({0:1, 3:1}, {0:1, 3:1}, [(0,1),(3,1)]),
            ({2:1, 3:1}, {0:1}, [(2,1),(3,-1)]),
            ({0:1}, {1:1, 3:-1}, [(1,1),(3,1)]),
            ({3:1}, {2:1, 0:-1}, [(0,1),(2,1)]),
            ({0:1, 1:1}, {3:1}, [(1,1),(0,-1)]),
            ({2:1, 0:-1}, {0:1, 1:1}, [(3,1)]),
            ({1:1, 3:-1}, {2:1, 3:1}, [(0,1),(1,1)]),
        ]
        
        for a_dict, b_dict, c_contribs in strassen_defs:
            c = offset
            u_coeffs = {}
            for a_idx, a_coef in a_dict.items():
                for (ar, ac, form_coef) in a_forms[a_idx]:
                    key = (ar, ac)
                    u_coeffs[key] = u_coeffs.get(key, 0) + a_coef * form_coef
            for (ar, ac), val in u_coeffs.items():
                U[ar*m+ac][c] += val
            
            v_coeffs = {}
            for b_idx, b_coef in b_dict.items():
                for (br, bc, form_coef) in b_forms[b_idx]:
                    key = (br, bc)
                    v_coeffs[key] = v_coeffs.get(key, 0) + b_coef * form_coef
            for (br, bc), val in v_coeffs.items():
                V[br*p+bc][c] += val
            
            for c_idx, c_coef in c_contribs:
                for (cr, cc, out_coef) in c_outputs[c_idx]:
                    W[cr*n+cc][c] += c_coef * out_coef
            
            offset += 1
        
        return offset
    
    # M1=(A00+A11)*(B00+B11): C00 and C11
    a1 = {0: [(0,0,1),(2,2,1)], 1: [(0,1,1),(2,3,1)], 2: [(1,0,1),(3,2,1)], 3: [(1,1,1),(3,3,1)]}
    b1 = {0: [(0,0,1),(2,2,1)], 1: [(0,1,1),(2,3,1)], 2: [(1,0,1),(3,2,1)], 3: [(1,1,1),(3,3,1)]}
    c1 = [
        [(0,0,1),(2,2,1)],  # c0
        [(0,1,1),(2,3,1)],  # c1
        [(1,0,1),(3,2,1)],  # c2
        [(1,1,1),(3,3,1)],  # c3
    ]
    col = add_block_strassen_v2(a1, b1, c1, col)
    
    # M2=(A10+A11)*B00: C10 gets +, C11 gets -
    a2 = {0: [(2,0,1),(0,0,1)], 1: [(2,1,1),(0,1,1)], 2: [(3,0,1),(1,0,1)], 3: [(3,1,1),(1,1,1)]}
    b2 = {0: [(0,0,1)], 1: [(0,1,1)], 2: [(1,0,1)], 3: [(1,1,1)]}
    c2 = [
        [(2,0,1),(2,2,-1)],  # c0
        [(2,1,1),(2,3,-1)],  # c1
        [(3,0,1),(3,2,-1)],  # c2
        [(3,1,1),(3,3,-1)],  # c3
    ]
    col = add_block_strassen_v2(a2, b2, c2, col)
    
    # M3=A00*(B01-B11): C01 gets +, C11 gets +
    a3 = {0: [(0,0,1)], 1: [(0,1,1)], 2: [(1,0,1)], 3: [(1,1,1)]}
    b3 = {0: [(0,2,1),(2,2,-1)], 1: [(0,3,1),(2,3,-1)], 2: [(1,2,1),(3,2,-1)], 3: [(1,3,1),(3,3,-1)]}
    c3 = [
        [(0,2,1),(2,2,1)],  # c0
        [(0,3,1),(2,3,1)],  # c1
        [(1,2,1),(3,2,1)],  # c2
        [(1,3,1),(3,3,1)],  # c3
    ]
    col = add_block_strassen_v2(a3, b3, c3, col)
    
    # M4=A11*(B10-B00): C00 gets +, C10 gets +
    a4 = {0: [(2,2,1)], 1: [(2,3,1)], 2: [(3,2,1)], 3: [(3,3,1)]}
    b4 = {0: [(2,0,1),(0,0,-1)], 1: [(2,1,1),(0,1,-1)], 2: [(3,0,1),(1,0,-1)], 3: [(3,1,1),(1,1,-1)]}
    c4 = [
        [(0,0,1),(2,0,1)],  # c0
        [(0,1,1),(2,1,1)],  # c1
        [(1,0,1),(3,0,1)],  # c2
        [(1,1,1),(3,1,1)],  # c3
    ]
    col = add_block_strassen_v2(a4, b4, c4, col)
    
    # M5=(A00+A01)*B11: C01 gets +, C00 gets -
    a5 = {0: [(0,0,1),(0,2,1)], 1: [(0,1,1),(0,3,1)], 2: [(1,0,1),(1,2,1)], 3: [(1,1,1),(1,3,1)]}
    b5 = {0: [(2,2,1)], 1: [(2,3,1)], 2: [(3,2,1)], 3: [(3,3,1)]}
    c5 = [
        [(0,2,1),(0,0,-1)],  # c0
        [(0,3,1),(0,1,-1)],  # c1
        [(1,2,1),(1,0,-1)],  # c2
        [(1,3,1),(1,1,-1)],  # c3
    ]
    col = add_block_strassen_v2(a5, b5, c5, col)
    
    # M6=(A10-A00)*(B00+B01): C11 gets +
    a6 = {0: [(2,0,1),(0,0,-1)], 1: [(2,1,1),(0,1,-1)], 2: [(3,0,1),(1,0,-1)], 3: [(3,1,1),(1,1,-1)]}
    b6 = {0: [(0,0,1),(0,2,1)], 1: [(0,1,1),(0,3,1)], 2: [(1,0,1),(1,2,1)], 3: [(1,1,1),(1,3,1)]}
    c6 = [
        [(2,2,1)],  # c0
        [(2,3,1)],  # c1
        [(3,2,1)],  # c2
        [(3,3,1)],  # c3
    ]
    col = add_block_strassen_v2(a6, b6, c6, col)
    
    # M7=(A01-A11)*(B10+B11): C00 gets +, C01 gets +
    a7 = {0: [(0,2,1),(2,2,-1)], 1: [(0,3,1),(2,3,-1)], 2: [(1,2,1),(3,2,-1)], 3: [(1,3,1),(3,3,-1)]}
    b7 = {0: [(2,0,1),(2,2,1)], 1: [(2,1,1),(2,3,1)], 2: [(3,0,1),(3,2,1)], 3: [(3,1,1),(3,3,1)]}
    c7 = [
        [(0,0,1),(0,2,1)],  # c0
        [(0,1,1),(0,3,1)],  # c1
        [(1,0,1),(1,2,1)],  # c2
        [(1,1,1),(1,3,1)],  # c3
    ]
    col = add_block_strassen_v2(a7, b7, c7, col)
    
    # Part 2: C[:, 4:9] = A * B[:, 4:9] (4x4 * 4x5) schoolbook, rank 20
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
