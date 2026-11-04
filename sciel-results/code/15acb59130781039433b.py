"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    rank = 72
    U = [[0] * rank for _ in range(n * m)]  # 16 x 72
    V = [[0] * rank for _ in range(m * p)]  # 36 x 72
    W = [[0] * rank for _ in range(p * n)]  # 36 x 72
    
    col_idx = 0
    
    def add_strassen_4x4x4(a_offset, b_offset, c_offset):
        nonlocal col_idx
        
        # A indices relative to a_offset
        # A11: 0, 1, 4, 5
        # A12: 2, 3, 6, 7
        # A21: 8, 9, 12, 13
        # A22: 10, 11, 14, 15
        
        # B indices relative to b_offset
        # B11: 0, 1, 9, 10
        # B12: 2, 3, 11, 12
        # B21: 4, 5, 13, 14
        # B22: 6, 7, 15, 16
        
        # C indices relative to c_offset
        # C11: 0, 1, 4, 5
        # C12: 2, 3, 6, 7
        # C21: 8, 9, 12, 13
        # C22: 10, 11, 14, 15

        # Define Strassen terms
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # Each M is 2x2.
        # We iterate through the 4 scalar products for each M.
        
        # Helper to add a rank 1 term
        def add_term(a_idx, b_idx, c_idx, coeff):
            nonlocal col_idx
            U[a_offset + a_idx][col_idx] = coeff
            V[b_offset + b_idx][col_idx] = 1
            W[c_offset + c_idx][col_idx] = 1
            col_idx += 1
            
        # We need to expand products like (A[0] + A[10]) * (B[0] + B[6])
        # This results in 4 terms:
        # A[0]*B[0], A[0]*B[6], A[10]*B[0], A[10]*B[6]
        
        # Let's pre-define the linear combinations for X and Y in each M_k
        # X is 2x2, Y is 2x2.
        # X[i][j] is a list of (a_idx, coeff)
        # Y[i][j] is a list of (b_idx, coeff)
        
        # M1
        X1 = [
            [(0, 1), (10, 1)],   # A11[0,0] + A22[0,0]
            [(1, 1), (11, 1)],   # A11[0,1] + A22[0,1]
            [(4, 1), (14, 1)],   # A11[1,0] + A22[1,0]
            [(5, 1), (15, 1)]    # A11[1,1] + A22[1,1]
        ]
        Y1 = [
            [(0, 1), (6, 1)],    # B11[0,0] + B22[0,0]
            [(1, 1), (7, 1)],    # B11[0,1] + B22[0,1]
            [(9, 1), (15, 1)],   # B11[1,0] + B22[1,0]
            [(10, 1), (16, 1)]   # B11[1,1] + B22[1,1]
        ]
        
        # M2
        X2 = [
            [(8, 1), (10, 1)],   # A21[0,0] + A22[0,0]
            [(9, 1), (11, 1)],   # A21[0,1] + A22[0,1]
            [(12, 1), (14, 1)],  # A21[1,0] + A22[1,0]
            [(13, 1), (15, 1)]   # A21[1,1] + A22[1,1]
        ]
        Y2 = [
            [(0, 1)],            # B11[0,0]
            [(1, 1)],            # B11[0,1]
            [(9, 1)],            # B11[1,0]
            [(10, 1)]            # B11[1,1]
        ]
        
        # M3
        X3 = [
            [(0, 1)],            # A11[0,0]
            [(1, 1)],            # A11[0,1]
            [(4, 1)],            # A11[1,0]
            [(5, 1)]             # A11[1,1]
        ]
        Y3 = [
            [(4, 1), (6, -1)],   # B21[0,0] - B22[0,0]
            [(5, 1), (7, -1)],   # B21[0,1] - B22[0,1]
            [(13, 1), (15, -1)], # B21[1,0] - B22[1,0]
            [(14, 1), (16, -1)]  # B21[1,1] - B22[1,1]
        ]
        
        # M4
        X4 = [
            [(10, 1)],           # A22[0,0]
            [(11, 1)],           # A22[0,1]
            [(14, 1)],           # A22[1,0]
            [(15, 1)]            # A22[1,1]
        ]
        Y4 = [
            [(2, 1), (0, -1)],   # B12[0,0] - B11[0,0]
            [(3, 1), (1, -1)],   # B12[0,1] - B11[0,1]
            [(11, 1), (9, -1)],  # B12[1,0] - B11[1,0]
            [(12, 1), (10, -1)]  # B12[1,1] - B11[1,1]
        ]
        
        # M5
        X5 = [
            [(0, 1), (2, 1)],    # A11[0,0] + A12[0,0]
            [(1, 1), (3, 1)],    # A11[0,1] + A12[0,1]
            [(4, 1), (6, 1)],    # A11[1,0] + A12[1,0]
            [(5, 1), (7, 1)]     # A11[1,1] + A12[1,1]
        ]
        Y5 = [
            [(6, 1)],            # B22[0,0]
            [(7, 1)],            # B22[0,1]
            [(15, 1)],           # B22[1,0]
            [(16, 1)]            # B22[1,1]
        ]
        
        # M6
        X6 = [
            [(8, 1), (0, -1)],   # A21[0,0] - A11[0,0]
            [(9, 1), (1, -1)],   # A21[0,1] - A11[0,1]
            [(12, 1), (4, -1)],  # A21[1,0] - A11[1,0]
            [(13, 1), (5, -1)]   # A21[1,1] - A11[1,1]
        ]
        Y6 = [
            [(0, 1), (2, 1)],    # B11[0,0] + B12[0,0]
            [(1, 1), (3, 1)],    # B11[0,1] + B12[0,1]
            [(9, 1), (11, 1)],   # B11[1,0] + B12[1,0]
            [(10, 1), (12, 1)]   # B11[1,1] + B12[1,1]
        ]
        
        # M7
        X7 = [
            [(2, 1), (10, -1)],  # A12[0,0] - A22[0,0]
            [(3, 1), (11, -1)],  # A12[0,1] - A22[0,1]
            [(6, 1), (14, -1)],  # A12[1,0] - A22[1,0]
            [(7, 1), (15, -1)]   # A12[1,1] - A22[1,1]
        ]
        Y7 = [
            [(4, 1), (6, 1)],    # B21[0,0] + B22[0,0]
            [(5, 1), (7, 1)],    # B21[0,1] + B22[0,1]
            [(13, 1), (15, 1)],  # B21[1,0] + B22[1,0]
            [(14, 1), (16, 1)]   # B21[1,1] + B22[1,1]
        ]

        # C contributions
        # C11 = M1 + M4 - M5 - M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Map (M_idx, r, c) to list of (C_block_idx, C_r, C_c, coeff)
        # C_block_idx: 0=C11, 1=C12, 2=C21, 3=C22
        # C_r, C_c: 0 or 1 within the 2x2 block
        
        def get_c_contribs(m_idx, r, c):
            contribs = []
            if m_idx == 1:
                if r == 0 and c == 0:
                    contribs.append((0, 0, 0, 1)) # C11[0,0]
                    contribs.append((3, 0, 0, 1)) # C22[0,0]
                elif r == 0 and c == 1:
                    contribs.append((0, 0, 1, 1)) # C11[0,1]
                    contribs.append((3, 0, 1, 1)) # C22[0,1]
                elif r == 1 and c == 0:
                    contribs.append((0, 1, 0, 1)) # C11[1,0]
                    contribs.append((3, 1, 0, 1)) # C22[1,0]
                elif r == 1 and c == 1:
                    contribs.append((0, 1, 1, 1)) # C11[1,1]
                    contribs.append((3, 1, 1, 1)) # C22[1,1]
            elif m_idx == 2:
                if r == 0 and c == 0:
                    contribs.append((2, 0, 0, 1)) # C21[0,0]
                    contribs.append((3, 0, 0, -1)) # C22[0,0]
                elif r == 0 and c == 1:
                    contribs.append((2, 0, 1, 1)) # C21[0,1]
                    contribs.append((3, 0, 1, -1)) # C22[0,1]
                elif r == 1 and c == 0:
                    contribs.append((2, 1, 0, 1)) # C21[1,0]
                    contribs.append((3, 1, 0, -1)) # C22[1,0]
                elif r == 1 and c == 1:
                    contribs.append((2, 1, 1, 1)) # C21[1,1]
                    contribs.append((3, 1, 1, -1)) # C22[1,1]
            elif m_idx == 3:
                if r == 0 and c == 0:
                    contribs.append((1, 0, 0, 1)) # C12[0,0]
                    contribs.append((3, 0, 0, 1)) # C22[0,0]
                elif r == 0 and c == 1:
                    contribs.append((1, 0, 1, 1)) # C12[0,1]
                    contribs.append((3, 0, 1, 1)) # C22[0,1]
                elif r == 1 and c == 0:
                    contribs.append((1, 1, 0, 1)) # C12[1,0]
                    contribs.append((3, 1, 0, 1)) # C22[1,0]
                elif r == 1 and c == 1:
                    contribs.append((1, 1, 1, 1)) # C12[1,1]
                    contribs.append((3, 1, 1, 1)) # C22[1,1]
            elif m_idx == 4:
                if r == 0 and c == 0:
                    contribs.append((0, 0, 0, 1)) # C11[0,0]
                    contribs.append((2, 0, 0, 1)) # C21[0,0]
                elif r == 0 and c == 1:
                    contribs.append((0, 0, 1, 1)) # C11[0,1]
                    contribs.append((2, 0, 1, 1)) # C21[0,1]
                elif r == 1 and c == 0:
                    contribs.append((0, 1, 0, 1)) # C11[1,0]
                    contribs.append((2, 1, 0, 1)) # C21[1,0]
                elif r == 1 and c == 1:
                    contribs.append((0, 1, 1, 1)) # C11[1,1]
                    contribs.append((2, 1, 1, 1)) # C21[1,1]
            elif m_idx == 5:
                if r == 0 and c == 0:
                    contribs.append((0, 0, 0, -1)) # C11[0,0]
                    contribs.append((1, 0, 0, 1)) # C12[0,0]
                elif r == 0 and c == 1:
                    contribs.append((0, 0, 1, -1)) # C11[0,1]
                    contribs.append((1, 0, 1, 1)) # C12[0,1]
                elif r == 1 and c == 0:
                    contribs.append((0, 1, 0, -1)) # C11[1,0]
                    contribs.append((1, 1, 0, 1)) # C12[1,0]
                elif r == 1 and c == 1:
                    contribs.append((0, 1, 1, -1)) # C11[1,1]
                    contribs.append((1, 1, 1, 1)) # C12[1,1]
            elif m_idx == 6:
                if r == 0 and c == 0:
                    contribs.append((3, 0, 0, 1)) # C22[0,0]
                elif r == 0 and c == 1:
                    contribs.append((3, 0, 1, 1)) # C22[0,1]
                elif r == 1 and c == 0:
                    contribs.append((3, 1, 0, 1)) # C22[1,0]
                elif r == 1 and c == 1:
                    contribs.append((3, 1, 1, 1)) # C22[1,1]
            elif m_idx == 7:
                if r == 0 and c == 0:
                    contribs.append((0, 0, 0, -1)) # C11[0,0]
                elif r == 0 and c == 1:
                    contribs.append((0, 0, 1, -1)) # C11[0,1]
                elif r == 1 and c == 0:
                    contribs.append((0, 1, 0, -1)) # C11[1,0]
                elif r == 1 and c == 1:
                    contribs.append((0, 1, 1, -1)) # C11[1,1]
            return contribs

        # Process each M
        for m_idx in range(1, 8):
            if m_idx == 1: X, Y = X1, Y1
            elif m_idx == 2: X, Y = X2, Y2
            elif m_idx == 3: X, Y = X3, Y3
            elif m_idx == 4: X, Y = X4, Y4
            elif m_idx == 5: X, Y = X5, Y5
            elif m_idx == 6: X, Y = X6, Y6
            elif m_idx == 7: X, Y = X7, Y7
            
            # Schoolbook 2x2
            # M[r][c] = X[r][0]Y[0][c] + X[r][1]Y[1][c]
            for r in range(2):
                for c in range(2):
                    # Term 1: X[r][0] * Y[0][c]
                    x_terms = X[r*2]
                    y_terms = Y[c] # Y[0][c] is index c in flat Y? 
                    # My X and Y are flattened 2x2.
                    # X[0]=X[0,0], X[1]=X[0,1], X[2]=X[1,0], X[3]=X[1,1]
                    # Y[0]=Y[0,0], Y[1]=Y[0,1], Y[2]=Y[1,0], Y[3]=Y[1,1]
                    
                    # M[r][c] = X[r,0]Y[0,c] + X[r,1]Y[1,c]
                    # X[r,0] is X[r*2]
                    # Y[0,c] is Y[c]
                    # X[r,1] is X[r*2+1]
                    # Y[1,c] is Y[2+c]
                    
                    # Product 1: X[r*2] * Y[c]
                    for a_idx, a_coeff in x_terms:
                        for b_idx, b_coeff in y_terms:
                            coeff = a_coeff * b_coeff
                            # Get C contributions
                            contribs = get_c_contribs(m_idx, r, c)
                            for cb, cr, cc, c_coeff in contribs:
                                # C index in local 4x4
                                # C11: 0,1,4,5 -> block 0
                                # C12: 2,3,6,7 -> block 1
                                # C21: 8,9,12,13 -> block 2
                                # C22: 10,11,14,15 -> block 3
                                # Index = cb*4 + cr*2 + cc
                                c_idx = cb*4 + cr*2 + cc
                                add_term(a_idx, b_idx, c_idx, coeff * c_coeff)
                    
                    # Product 2: X[r*2+1] * Y[2+c]
                    x_terms2 = X[r*2+1]
                    y_terms2 = Y[2+c]
                    for a_idx, a_coeff in x_terms2:
                        for b_idx, b_coeff in y_terms2:
                            coeff = a_coeff * b_coeff
                            contribs = get_c_contribs(m_idx, r, c)
                            for cb, cr, cc, c_coeff in contribs:
                                c_idx = cb*4 + cr*2 + cc
                                add_term(a_idx, b_idx, c_idx, coeff * c_coeff)

    def add_schoolbook_4x4x1(a_offset, b_offset, c_offset):
        nonlocal col_idx
        # A is 4x4, B is 4x1, C is 4x1
        # C[i] = sum_j A[i,j] B[j]
        for i in range(4):
            for j in range(4):
                a_idx = i*4 + j
                b_idx = j # B is 4x1, so index j
                c_idx = i # C is 4x1, so index i
                U[a_offset + a_idx][col_idx] = 1
                V[b_offset + b_idx][col_idx] = 1
                W[c_offset + c_idx][col_idx] = 1
                col_idx += 1

    # 1. A (4x4) * B_L (4x4, cols 0-3)
    # A offset 0
    # B offset 0 (cols 0-3)
    # C offset 0 (cols 0-3)
    add_strassen_4x4x4(0, 0, 0)
    
    # 2. A (4x4) * B_R1 (4x4, cols 4-7)
    # A offset 0
    # B offset 4 (cols 4-7)
    # C offset 4 (cols 4-7)
    add_strassen_4x4x4(0, 4, 4)
    
    # 3. A (4x4) * B_R2 (4x1, col 8)
    # A offset 0
    # B offset 8 (col 8)
    # C offset 8 (col 8)
    add_schoolbook_4x4x1(0, 8, 8)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
