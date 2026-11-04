import numpy as np

DIMENSIONS = [4, 4, 6]

def get_strassen_2x2x2():
    # Returns U, V, W for 2x2x2 Strassen
    # n=2, m=2, p=2
    # U shape (4, 7), V shape (4, 7), W shape (4, 7)
    U = np.zeros((4, 7), dtype=int)
    V = np.zeros((4, 7), dtype=int)
    W = np.zeros((4, 7), dtype=int)
    
    # Term 1: P1 = (A00+A11)(B00+B11)
    # C00 += P1, C11 += P1
    U[0, 0] = 1; U[3, 0] = 1 # A00, A11
    V[0, 0] = 1; V[3, 0] = 1 # B00, B11
    W[0, 0] = 1; W[3, 0] = 1 # C00, C11
    
    # Term 2: P2 = (A10+A11)B00
    # C10 += P2
    U[2, 1] = 1; U[3, 1] = 1 # A10, A11
    V[0, 1] = 1              # B00
    W[1, 1] = 1              # C10 (i=1, k=0 -> idx 1*2+0=2? No, W index is k*n+i. n=2. k=0, i=1 -> 0*2+1=1)
    # Wait, W index for C10: i=1, k=0. Index = 0*2 + 1 = 1. Correct.
    
    # Term 3: P3 = A00(B01-B11)
    # C01 += P3, C11 += P3
    U[0, 2] = 1              # A00
    V[1, 2] = 1; V[3, 2] = -1 # B01, B11
    W[2, 2] = 1; W[3, 2] = 1  # C01 (k=1, i=0 -> 1*2+0=2), C11 (k=1, i=1 -> 1*2+1=3)
    
    # Term 4: P4 = A11(B10-B00)
    # C00 += P4, C10 += P4
    U[3, 3] = 1              # A11
    V[2, 3] = 1; V[0, 3] = -1 # B10, B00
    W[0, 3] = 1; W[1, 3] = 1  # C00 (k=0, i=0 -> 0), C10 (k=0, i=1 -> 1)
    
    # Term 5: P5 = (A00+A01)B11
    # C00 -= P5, C01 += P5
    U[0, 4] = 1; U[1, 4] = 1 # A00, A01
    V[3, 4] = 1              # B11
    W[0, 4] = -1; W[2, 4] = 1 # C00 (0), C01 (2)
    
    # Term 6: P6 = (A10-A00)(B00+B01)
    # C11 += P6
    U[2, 5] = 1; U[0, 5] = -1 # A10, A00
    V[0, 5] = 1; V[1, 5] = 1  # B00, B01
    W[3, 5] = 1              # C11 (3)
    
    # Term 7: P7 = (A01-A11)(B10+B11)
    # C00 += P7
    U[1, 6] = 1; U[3, 6] = -1 # A01, A11
    V[2, 6] = 1; V[3, 6] = 1  # B10, B11
    W[0, 6] = 1              # C00 (0)
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    # We will build rank 49 for 4x4x4 and rank 28 for 4x4x2
    # Total rank 77
    
    U = [[0] * 77 for _ in range(n * m)] # 16x77
    V = [[0] * 77 for _ in range(m * p)] # 24x77
    W = [[0] * 77 for _ in range(p * n)] # 24x77
    
    # Part 1: 4x4x4 (cols 0-3 of B, cols 0-3 of C)
    # Rank 49
    U_s, V_s, W_s = get_strassen_2x2x2()
    
    # Top level Strassen terms for 4x4x4
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1)
    # B blocks: B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1)
    # C blocks: C00 (0,0), C01 (0,1), C10 (1,0), C11 (1,1)
    
    # Term 1: (A00+A11)(B00+B11) -> C00 += P, C11 += P
    # X = A00+A11, Y = B00+B11
    # Sub-terms of X*Y
    for r in range(7):
        # U part: X = A00+A11
        # A00 is at (0,0), A11 is at (2,2) in 4x4
        # Local U indices for A00: 0,1,2,3? No, 2x2 block.
        # A00 covers i=0,1; j=0,1.
        # A11 covers i=2,3; j=2,3.
        
        # We need to map the 2x2x2 decomposition of X*Y to the global indices.
        # X is a 2x2 matrix formed by A00+A11.
        # Y is a 2x2 matrix formed by B00+B11.
        # The product P is a 2x2 matrix.
        # P contributes to C00 and C11.
        
        # This logic is getting complex. I will use a helper to add a 2x2x2 block.
        # But here X and Y are sums of blocks.
        
        pass # Skip detailed implementation for now, assume I can generate it.

    # Given the complexity, I will provide a known rank 77 solution if I can find it, 
    # or stick to a simpler improvement if possible.
    # But Strassen is the way.
    
    # I will implement the 49-rank 4x4x4 and 28-rank 4x4x2 by explicitly listing the terms.
    # This is tedious but doable.
    
    # For the sake of this response, I will implement the 49-rank 4x4x4 part correctly.
    # And the 28-rank 4x4x2 part correctly.
    
    # Helper to add a 2x2x2 Strassen term to global matrices
    def add_2x2x2_term(U_g, V_g, W_g, rank_idx, 
                       a_blocks, b_blocks, c_blocks, 
                       n_g, m_g, p_g):
        # a_blocks: list of (i_off, j_off, coeff)
        # b_blocks: list of (j_off, k_off, coeff)
        # c_blocks: list of (i_off, k_off, coeff)
        
        U_s, V_s, W_s = get_strassen_2x2x2()
        
        for r_sub in range(7):
            r_global = rank_idx + r_sub
            
            # U part
            for i_l in range(2):
                for j_l in range(2):
                    val_u = U_s[i_l * 2 + j_l, r_sub]
                    if val_u != 0:
                        for i_b, j_b, c_a in a_blocks:
                            i_g = i_b * 2 + i_l
                            j_g = j_b * 2 + j_l
                            U_g[i_g * m_g + j_g][r_global] += val_u * c_a
            
            # V part
            for j_l in range(2):
                for k_l in range(2):
                    val_v = V_s[j_l * 2 + k_l, r_sub]
                    if val_v != 0:
                        for j_b, k_b, c_b in b_blocks:
                            j_g = j_b * 2 + j_l
                            k_g = k_b * 2 + k_l
                            V_g[j_g * p_g + k_g][r_global] += val_v * c_b
            
            # W part
            for k_l in range(2):
                for i_l in range(2):
                    val_w = W_s[k_l * 2 + i_l, r_sub]
                    if val_w != 0:
                        for i_c, k_c, c_c in c_blocks:
                            i_g = i_c * 2 + i_l
                            k_g = k_c * 2 + k_l
                            W_g[k_g * n_g + i_g][r_global] += val_w * c_c

    # 4x4x4 Part
    # n=4, m=4, p=4
    # Term 1: X=A00+A11, Y=B00+B11, C=C00+C11
    a_blocks = [(0,0,1), (1,1,1)]
    b_blocks = [(0,0,1), (1,1,1)]
    c_blocks = [(0,0,1), (1,1,1)]
    add_2x2x2_term(U, V, W, 0, a_blocks, b_blocks, c_blocks, 4, 4, 4)
    
    # Term 2: X=A10+A11, Y=B00, C=C10
    a_blocks = [(1,0,1), (1,1,1)]
    b_blocks = [(0,0,1)]
    c_blocks = [(1,0,1)]
    add_2x2x2_term(U, V, W, 7, a_blocks, b_blocks, c_blocks, 4, 4, 4)
    
    # Term 3: X=A00, Y=B01-B11, C=C01+C11
    a_blocks = [(0,0,1)]
    b_blocks = [(0,1,1), (1,1,-1)]
    c_blocks = [(0,1,1), (1,1,1)]
    add_2x2x2_term(U, V, W, 14, a_blocks, b_blocks, c_blocks, 4, 4, 4)
    
    # Term 4: X=A11, Y=B10-B00, C=C00+C10
    a_blocks = [(1,1,1)]
    b_blocks = [(1,0,1), (0,0,-1)]
    c_blocks = [(0,0,1), (1,0,1)]
    add_2x2x2_term(U, V, W, 21, a_blocks, b_blocks, c_blocks, 4, 4, 4)
    
    # Term 5: X=A00+A01, Y=B11, C=C00-C01
    a_blocks = [(0,0,1), (0,1,1)]
    b_blocks = [(1,1,1)]
    c_blocks = [(0,0,-1), (0,1,1)]
    add_2x2x2_term(U, V, W, 28, a_blocks, b_blocks, c_blocks, 4, 4, 4)
    
    # Term 6: X=A10-A00, Y=B00+B01, C=C11
    a_blocks = [(1,0,1), (0,0,-1)]
    b_blocks = [(0,0,1), (0,1,1)]
    c_blocks = [(1,1,1)]
    add_2x2x2_term(U, V, W, 35, a_blocks, b_blocks, c_blocks, 4, 4, 4)
    
    # Term 7: X=A01-A11, Y=B10+B11, C=C00
    a_blocks = [(0,1,1), (1,1,-1)]
    b_blocks = [(1,0,1), (1,1,1)]
    c_blocks = [(0,0,1)]
    add_2x2x2_term(U, V, W, 42, a_blocks, b_blocks, c_blocks, 4, 4, 4)
    
    # 4x4x2 Part (cols 4-5 of B, cols 4-5 of C)
    # B is 4x2. Blocks B1 (rows 0-1), B2 (rows 2-3).
    # C is 4x2. Blocks C1 (rows 0-1), C2 (rows 2-3).
    # Terms:
    # 1. A00 B1 -> C1
    # 2. A01 B2 -> C1
    # 3. A10 B1 -> C2
    # 4. A11 B2 -> C2
    
    # Note: B1 is 2x2, B2 is 2x2.
    # In global B (4x6), B1 is at j=0,1; k=4,5.
    # B2 is at j=2,3; k=4,5.
    # In global C (6x4), C1 is at i=0,1; k=4,5.
    # C2 is at i=2,3; k=4,5.
    
    # Term 8: A00 B1 -> C1
    a_blocks = [(0,0,1)]
    b_blocks = [(0,2,1)] # j=0, k=2 (local k for 4x2 is 0,1? No, global k is 4,5. Local k in 2x2 is 0,1. Global k = 4 + k_l. So k_b=2 means global k=4+0? No.
    # Let's adjust add_2x2x2_term to handle global offsets properly.
    # The helper assumes local 2x2 blocks.
    # For 4x4x2, the "blocks" are not 2x2 in the same way.
    # I will hardcode the 28 terms for 4x4x2 to be safe.
    
    # Actually, the 4x4x2 is just 4 independent 2x2x2 multiplications.
    # 1. A[0:2, 0:2] * B[0:2, 4:6] -> C[0:2, 4:6]
    # 2. A[0:2, 2:4] * B[2:4, 4:6] -> C[0:2, 4:6]
    # 3. A[2:4, 0:2] * B[0:2, 4:6] -> C[2:4, 4:6]
    # 4. A[2:4, 2:4] * B[2:4, 4:6] -> C[2:4, 4:6]
    
    # I will use the helper but with appropriate offsets.
    # The helper adds to U[i_g * m + j_g].
    # For 4x4x2, m=4, p=6.
    # B is 4x6.
    # B1 is rows 0-1, cols 4-5.
    # In the helper, b_blocks are (j_b, k_b).
    # j_b is block index (0 or 1). k_b is block index (0 or 1).
    # Global j = j_b * 2 + j_l.
    # Global k = k_b * 2 + k_l + k_offset.
    # I need to modify the helper to accept k_offset for B and C.
    
    def add_2x2x2_term_offset(U_g, V_g, W_g, rank_idx, 
                       a_blocks, b_blocks, c_blocks, 
                       n_g, m_g, p_g, k_offset_b, k_offset_c):
        U_s, V_s, W_s = get_strassen_2x2x2()
        
        for r_sub in range(7):
            r_global = rank_idx + r_sub
            
            for i_l in range(2):
                for j_l in range(2):
                    val_u = U_s[i_l * 2 + j_l, r_sub]
                    if val_u != 0:
                        for i_b, j_b, c_a in a_blocks:
                            i_g = i_b * 2 + i_l
                            j_g = j_b * 2 + j_l
                            U_g[i_g * m_g + j_g][r_global] += val_u * c_a
            
            for j_l in range(2):
                for k_l in range(2):
                    val_v = V_s[j_l * 2 + k_l, r_sub]
                    if val_v != 0:
                        for j_b, k_b, c_b in b_blocks:
                            j_g = j_b * 2 + j_l
                            k_g = k_b * 2 + k_l + k_offset_b
                            V_g[j_g * p_g + k_g][r_global] += val_v * c_b
            
            for k_l in range(2):
                for i_l in range(2):
                    val_w = W_s[k_l * 2 + i_l, r_sub]
                    if val_w != 0:
                        for i_c, k_c, c_c in c_blocks:
                            i_g = i_c * 2 + i_l
                            k_g = k_c * 2 + k_l + k_offset_c
                            W_g[k_g * n_g + i_g][r_global] += val_w * c_c

    # 4x4x2 Terms
    # k_offset_b = 4, k_offset_c = 4
    
    # 1. A00 B1 -> C1
    a_blocks = [(0,0,1)]
    b_blocks = [(0,0,1)] # B1 is j=0, k=4,5. j_b=0, k_b=0.
    c_blocks = [(0,0,1)] # C1 is i=0, k=4,5. i_c=0, k_c=0.
    add_2x2x2_term_offset(U, V, W, 49, a_blocks, b_blocks, c_blocks, 4, 4, 6, 4, 4)
    
    # 2. A01 B2 -> C1
    a_blocks = [(0,1,1)]
    b_blocks = [(1,0,1)] # B2 is j=2, k=4,5. j_b=1, k_b=0.
    c_blocks = [(0,0,1)]
    add_2x2x2_term_offset(U, V, W, 56, a_blocks, b_blocks, c_blocks, 4, 4, 6, 4, 4)
    
    # 3. A10 B1 -> C2
    a_blocks = [(1,0,1)]
    b_blocks = [(0,0,1)]
    c_blocks = [(1,0,1)] # C2 is i=2, k=4,5. i_c=1, k_c=0.
    add_2x2x2_term_offset(U, V, W, 63, a_blocks, b_blocks, c_blocks, 4, 4, 6, 4, 4)
    
    # 4. A11 B2 -> C2
    a_blocks = [(1,1,1)]
    b_blocks = [(1,0,1)]
    c_blocks = [(1,0,1)]
    add_2x2x2_term_offset(U, V, W, 70, a_blocks, b_blocks, c_blocks, 4, 4, 6, 4, 4)
    
    return {"U": U, "V": V, "W": W}

