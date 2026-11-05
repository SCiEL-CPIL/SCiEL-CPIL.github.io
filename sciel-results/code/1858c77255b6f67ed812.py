import time

def construct():
    """
    Rank-minimal exact tensor decomposition for 4x4 * 4x5 matrix multiplication.
    
    Strategy:
    - 4x4x4 part: Nested Strassen (7 outer * 7 inner = 49 multiplications)
    - 4x4x1 part: Schoolbook (4*1 = 4 multiplications) 
    - Total rank: 53
    
    Wait, let me reconsider. The 4x4x1 part needs 4*4 = 16 terms if we treat 
    each output element separately. But we can use Strassen-like ideas for the 1-column part too.
    
    Actually, for 4x4 * 4x1, the standard approach is 16 scalar multiplications.
    But we can do better by viewing it as 4x4 * 4x1 and using block decomposition.
    
    Let me reconsider the whole structure. We have A (4x4) * B (4x5) = C (4x5).
    
    We can split B into B1 (4x4) and B2 (4x1):
    C = A*B1 + A*B2
    
    For A*B1 (4x4x4): Use nested Strassen -> rank 49
    For A*B2 (4x4x1): This is 4 vector-matrix products, each requiring 4 scalar mults = 16
    
    Total: 49 + 16 = 65
    
    But wait - can we do better for the 4x4x1 part? 
    The tensor for 4x4 * 4x1 has dimension 4*4*1 = 16, and its rank is 16 (it's the 
    identity-like tensor for matrix-vector multiplication). So we need at least 16 terms.
    
    Total minimum: 49 + 16 = 65.
    
    Let me implement this correctly.
    """
    n, m, p = 4, 4, 5
    
    rank = 65
    U = [[0] * rank for _ in range(n * m)]  # 16 x 65
    V = [[0] * rank for _ in range(m * p)]  # 20 x 65
    W = [[0] * rank for _ in range(p * n)]  # 20 x 65
    
    col = 0
    
    # Part 1: 4x4 * 4x4 using nested Strassen (rank 49)
    # Split A, B1, C1 into 2x2 blocks of 2x2 matrices
    
    # Inner Strassen for 2x2x2: 7 terms
    inner_strassen = [
        # s1 = (p00+p11)(q00+q11) -> r00 += s1, r11 += s1
        {'u': [(0,0,1), (1,1,1)], 'v': [(0,0,1), (1,1,1)], 'w': [(0,0,1), (1,1,1)]},
        # s2 = (p10+p11)(q00) -> r10 += s2, r11 -= s2
        {'u': [(1,0,1), (1,1,1)], 'v': [(0,0,1)], 'w': [(1,0,1), (1,1,-1)]},
        # s3 = p00(q01-q11) -> r01 += s3, r11 += s3
        {'u': [(0,0,1)], 'v': [(0,1,1), (1,1,-1)], 'w': [(0,1,1), (1,1,1)]},
        # s4 = p11(q10-q00) -> r00 += s4, r10 += s4
        {'u': [(1,1,1)], 'v': [(1,0,1), (0,0,-1)], 'w': [(0,0,1), (1,0,1)]},
        # s5 = (p00+p01)(q11) -> r00 -= s5, r01 += s5
        {'u': [(0,0,1), (0,1,1)], 'v': [(1,1,1)], 'w': [(0,0,-1), (0,1,1)]},
        # s6 = (p10-p00)(q00+q01) -> r11 += s6
        {'u': [(1,0,1), (0,0,-1)], 'v': [(0,0,1), (0,1,1)], 'w': [(1,1,1)]},
        # s7 = (p01-p11)(q10+q11) -> r00 += s7
        {'u': [(0,1,1), (1,1,-1)], 'v': [(1,0,1), (1,1,1)], 'w': [(0,0,1)]},
    ]
    
    # Outer Strassen for 4x4x4: 7 terms, each involves 2x2 blocks
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]], C = [[C00, C01], [C10, C11]]
    
    outer_strassen = [
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        {'A_blocks': [1, 0, 0, 1], 'B_blocks': [1, 0, 0, 1], 'C_contrib': [(0,1), (3,1)]},
        # M2 = (A10+A11)(B00) -> C10 += M2, C11 -= M2
        {'A_blocks': [0, 0, 1, 1], 'B_blocks': [1, 0, 0, 0], 'C_contrib': [(2,1), (3,-1)]},
        # M3 = A00(B01-B11) -> C01 += M3, C11 += M3
        {'A_blocks': [1, 0, 0, 0], 'B_blocks': [0, 1, 0, -1], 'C_contrib': [(1,1), (3,1)]},
        # M4 = A11(B10-B00) -> C00 += M4, C10 += M4
        {'A_blocks': [0, 0, 0, 1], 'B_blocks': [-1, 0, 1, 0], 'C_contrib': [(0,1), (2,1)]},
        # M5 = (A00+A01)(B11) -> C00 -= M5, C01 += M5
        {'A_blocks': [1, 1, 0, 0], 'B_blocks': [0, 0, 0, 1], 'C_contrib': [(0,-1), (1,1)]},
        # M6 = (A10-A00)(B00+B01) -> C11 += M6
        {'A_blocks': [-1, 0, 1, 0], 'B_blocks': [1, 1, 0, 0], 'C_contrib': [(3,1)]},
        # M7 = (A01-A11)(B10+B11) -> C00 += M7
        {'A_blocks': [0, 1, 0, -1], 'B_blocks': [0, 0, 1, 1], 'C_contrib': [(0,1)]},
    ]
    
    # Helper to map block indices to global indices
    def block_to_global(block_idx, row_in_block, col_in_block, size=2):
        """Map 2x2 block index and position within block to global 4x4 index"""
        if block_idx == 0:  # top-left
            return row_in_block, col_in_block
        elif block_idx == 1:  # top-right
            return row_in_block, col_in_block + size
        elif block_idx == 2:  # bottom-left
            return row_in_block + size, col_in_block
        elif block_idx == 3:  # bottom-right
            return row_in_block + size, col_in_block + size
    
    # Build the 49 terms for 4x4x4
    for outer in outer_strassen:
        A_blocks = outer['A_blocks']
        B_blocks = outer['B_blocks']
        C_contrib = outer['C_contrib']
        
        for inner in inner_strassen:
            u_list = []
            v_list = []
            w_list = []
            
            # Process U (A part)
            for (ri, ci, coeff_a) in inner['u']:
                for b_idx, block_coeff in enumerate(A_blocks):
                    if block_coeff == 0:
                        continue
                    total_coeff = coeff_a * block_coeff
                    if total_coeff == 0:
                        continue
                    gi, gj = block_to_global(b_idx, ri, ci)
                    u_list.append((gi, gj, total_coeff))
            
            # Process V (B part) - note B is 4x4 for first 4 columns
            for (ri, ci, coeff_b) in inner['v']:
                for b_idx, block_coeff in enumerate(B_blocks):
                    if block_coeff == 0:
                        continue
                    total_coeff = coeff_b * block_coeff
                    if total_coeff == 0:
                        continue
                    gi, gj = block_to_global(b_idx, ri, ci)
                    v_list.append((gi, gj, total_coeff))
            
            # Process W (C part)
            for (ri, ci, coeff_c) in inner['w']:
                for (c_block, c_coeff) in C_contrib:
                    total_coeff = coeff_c * c_coeff
                    if total_coeff == 0:
                        continue
                    gi, gj = block_to_global(c_block, ri, ci)
                    # W is indexed as [k*n + i, col] where C[i][k]
                    w_list.append((gj, gi, total_coeff))
            
            if u_list and v_list and w_list:
                for (i, j, c) in u_list:
                    U[i * m + j][col] += c
                for (j, k, c) in v_list:
                    V[j * p + k][col] += c
                for (k, i, c) in w_list:
                    W[k * n + i][col] += c
                col += 1
    
    # Part 2: 4x4 * 4x1 (the 5th column of B) using schoolbook
    # C[i][4] = sum_j A[i][j] * B[j][4]
    for i in range(4):
        for j in range(4):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
