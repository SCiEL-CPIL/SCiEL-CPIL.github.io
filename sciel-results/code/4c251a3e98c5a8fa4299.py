import time

def construct():
    n, m, p = 4, 4, 6
    
    # Helper to generate Strassen 2x2x2 decomposition (Rank 7)
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    def strassen_2x2():
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A11+A22)(B11+B22)
        # A11=0, A22=1, B11=0, B22=1
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1; W[1][0] = -1; W[2][0] = 1
        
        # M2 = (A21+A22)B11
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[0][1] = 1; W[2][1] = 1
        
        # M3 = A11(B12-B22)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = A22(B21-B11)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        # M5 = (A11+A12)B22
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[1][4] = -1; W[3][4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[0][5] = -1; W[1][5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[2][6] = -1; W[3][6] = 1
        
        return U, V, W

    # Build 4x4x4 Strassen (Rank 49)
    # A (4x4) split into 4 blocks of 2x2.
    # B (4x4) split into 4 blocks of 2x2.
    # C (4x4) split into 4 blocks of 2x2.
    # We need 7 "super" multiplications of 2x2 matrices.
    # Each super multiplication uses 7 scalar multiplications.
    
    U_4x4 = []
    V_4x4 = []
    W_4x4 = []
    
    # Indices for 4x4 matrix: i*4+j
    # Block (bi, bj) corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
    # U index: i*4+j
    # V index: j*4+k
    # W index: k*4+i
    
    # Strassen formulas for 2x2 blocks:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # Block indices:
    # A11: (0,0), A12: (0,1), A21: (1,0), A22: (1,1)
    # B11: (0,0), B12: (0,1), B21: (1,0), B22: (1,1)
    
    # We will generate the 7 terms for the block level, and for each term,
    # we will instantiate the 2x2 Strassen decomposition.
    
    base_U, base_V, base_W = strassen_2x2()
    
    # Term definitions: (A_blocks, B_blocks, C_blocks, signs)
    # A_blocks: list of (block_idx, sign)
    # B_blocks: list of (block_idx, sign)
    # C_blocks: list of (block_idx, sign)
    
    terms = [
        # M1
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,-1), (0,1,1), (1,0,1)]),
        # M2
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (0,0,1)]), # Wait, C21 = M2 + M4, C11 = M1 - M5 + M7 + M4?
        # Let's re-verify Strassen C formulas:
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
    ]
    
    # Correct Strassen C mapping:
    # M1 -> C11(+), C22(+)
    # M2 -> C21(+), C22(-)
    # M3 -> C12(+), C22(+)
    # M4 -> C11(+), C21(+)
    # M5 -> C11(-), C12(+)
    # M6 -> C22(+)
    # M7 -> C11(+)
    
    # We will iterate over the 7 M-terms.
    # For each M-term, we define the A-block combination and B-block combination.
    # Then we use the base 2x2 Strassen to expand it into 7 scalar terms.
    
    m_definitions = [
        # (A_blocks, B_blocks)
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]), # M1
        ([(1,0,1), (1,1,1)], [(0,0,1)]),           # M2
        ([(0,0,1)], [(0,1,1), (1,1,-1)]),          # M3
        ([(1,1,1)], [(1,0,1), (0,0,-1)]),          # M4
        ([(0,0,1), (0,1,1)], [(1,1,1)]),           # M5
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]), # M6
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)])  # M7
    ]
    
    c_definitions = [
        [(0,0,1), (1,1,1)], # M1
        [(1,0,1), (1,1,-1)], # M2
        [(0,1,1), (1,1,1)], # M3
        [(0,0,1), (1,0,1)], # M4
        [(0,0,-1), (0,1,1)], # M5
        [(1,1,1)], # M6
        [(0,0,1)] # M7
    ]
    
    for m_idx in range(7):
        a_blocks = m_definitions[m_idx][0]
        b_blocks = m_definitions[m_idx][1]
        c_blocks = c_definitions[m_idx]
        
        # For each of the 7 scalar terms in the base 2x2 Strassen:
        for s_idx in range(7):
            # Create new U, V, W rows for this scalar term
            new_U = [0]*16
            new_V = [0]*16
            new_W = [0]*16
            
            # U (A part):
            for (bi, bj, sign) in a_blocks:
                # Block (bi, bj) of A corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
                # Indices in A: i*4+j
                for di in range(2):
                    for dj in range(2):
                        i = 2*bi + di
                        j = 2*bj + dj
                        a_idx = i*4 + j
                        # base_U[di*2+dj][s_idx] is the coefficient for A[di,dj] in this scalar term
                        coeff = base_U[di*2+dj][s_idx]
                        if coeff != 0:
                            new_U[a_idx] += sign * coeff
            
            # V (B part):
            for (bi, bj, sign) in b_blocks:
                # Block (bi, bj) of B corresponds to rows 2*bi..2*bi+1, cols 2*bj..2*bj+1
                # Indices in B: j*4+k (row j, col k)
                for di in range(2):
                    for dj in range(2):
                        j = 2*bi + di
                        k = 2*bj + dj
                        b_idx = j*4 + k
                        coeff = base_V[di*2+dj][s_idx]
                        if coeff != 0:
                            new_V[b_idx] += sign * coeff
            
            # W (C part):
            for (ci, cj, sign) in c_blocks:
                # Block (ci, cj) of C corresponds to rows 2*ci..2*ci+1, cols 2*cj..2*cj+1
                # Indices in C: k*4+i (row k, col i) -- Wait, W[k*n+i]
                # n=4. W index = k*4 + i.
                # C block (ci, cj) means row block ci, col block cj.
                # Row index in C: k = 2*ci + dk
                # Col index in C: i = 2*cj + di
                for dk in range(2):
                    for di in range(2):
                        k = 2*ci + dk
                        i = 2*cj + di
                        c_idx = k*4 + i
                        coeff = base_W[dk*2+di][s_idx]
                        if coeff != 0:
                            new_W[c_idx] += sign * coeff
            
            U_4x4.append(new_U)
            V_4x4.append(new_V)
            W_4x4.append(new_W)
    
    # Now handle 4x4x2 part (Rank 28)
    # A (4x4), B_right (4x2), C_right (4x2)
    # B_right cols 0,1.
    # Split B_right into B_top (2x2, rows 0-1) and B_bot (2x2, rows 2-3).
    # C_right into C_top (2x2, rows 0-1) and C_bot (2x2, rows 2-3).
    # C_top = A11 B_top + A12 B_bot
    # C_bot = A21 B_top + A22 B_bot
    
    U_4x2 = []
    V_4x2 = []
    W_4x2 = []
    
    # 4 products:
    # P1: A11 * B_top -> C_top
    # P2: A12 * B_bot -> C_top
    # P3: A21 * B_top -> C_bot
    # P4: A22 * B_bot -> C_bot
    
    products = [
        # (A_block, B_block, C_block)
        ((0,0), (0,0), (0,0)), # P1: A11 (0,0), B_top (rows 0-1 of B_right), C_top (rows 0-1 of C_right)
        ((0,1), (1,0), (0,0)), # P2: A12 (0,1), B_bot (rows 2-3 of B_right), C_top
        ((1,0), (0,0), (1,0)), # P3: A21 (1,0), B_top, C_bot (rows 2-3)
        ((1,1), (1,0), (1,0))  # P4: A22 (1,1), B_bot, C_bot
    ]
    
    for (ab, bb, cb) in products:
        # ab: (bi, bj) block of A
        # bb: (bi, bj) block of B_right. B_right is 4x2.
        # If bb[0]==0, rows 0-1. If bb[0]==1, rows 2-3.
        # bb[1] is always 0 (only 1 col block of width 2).
        # cb: (ci, cj) block of C_right.
        
        for s_idx in range(7):
            new_U = [0]*16
            new_V = [0]*24 # B is 4x6, so 24 entries. B_right is cols 4,5.
            new_W = [0]*24 # C is 4x6, so 24 entries. C_right is cols 4,5.
            
            # U part: A block ab
            bi, bj = ab
            for di in range(2):
                for dj in range(2):
                    i = 2*bi + di
                    j = 2*bj + dj
                    a_idx = i*4 + j
                    coeff = base_U[di*2+dj][s_idx]
                    if coeff != 0:
                        new_U[a_idx] = coeff
            
            # V part: B_right block bb
            # B_right is the right half of B (4x6).
            # B indices: row * 6 + col.
            # B_right rows: if bb[0]==0, rows 0-1. If bb[0]==1, rows 2-3.
            # B_right cols: 4-5.
            b_row_offset = 2 * bb[0]
            for di in range(2):
                for dj in range(2):
                    j = b_row_offset + di
                    k = 4 + dj # Cols 4,5
                    b_idx = j*6 + k
                    coeff = base_V[di*2+dj][s_idx]
                    if coeff != 0:
                        new_V[b_idx] = coeff
            
            # W part: C_right block cb
            # C_right is the right half of C (4x6).
            # C indices: col * 4 + row? No, W[k*n+i] -> row k, col i.
            # C_right rows: if cb[0]==0, rows 0-1. If cb[0]==1, rows 2-3.
            # C_right cols: 4-5.
            c_row_offset = 2 * cb[0]
            for dk in range(2):
                for di in range(2):
                    k = c_row_offset + dk
                    i = 4 + di # Cols 4,5
                    c_idx = k*4 + i
                    coeff = base_W[dk*2+di][s_idx]
                    if coeff != 0:
                        new_W[c_idx] = coeff
            
            U_4x2.append(new_U)
            V_4x2.append(new_V)
            W_4x2.append(new_W)
    
    # Combine
    U_final = U_4x4 + U_4x2
    V_final = []
    W_final = []
    
    # V_4x4 is 16 entries (4x4 B). We need to extend to 24 (4x6 B).
    for v in V_4x4:
        V_final.append(v + [0]*8) # Pad with 0s for cols 4,5
    
    V_final += V_4x2
    
    # W_4x4 is 16 entries (4x4 C). We need to extend to 24 (4x6 C).
    for w in W_4x4:
        W_final.append(w + [0]*8) # Pad with 0s for cols 4,5
        
    W_final += W_4x2
    
    return {"U": U_final, "V": V_final, "W": W_final}
