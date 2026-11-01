"""Rewritten program for matrix multiplication 2,2,9 tensor decomposition using Strassen blocks."""
DIMENSIONS = [2, 2, 9]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 9
    
    # We partition the 9 columns of B into four blocks of size 2 and one block of size 1.
    # Block 0: cols 0,1 (Rank 7)
    # Block 1: cols 2,3 (Rank 7)
    # Block 2: cols 4,5 (Rank 7)
    # Block 3: cols 6,7 (Rank 7)
    # Block 4: cols 8   (Rank 4)
    # Total Rank = 4*7 + 4 = 32
    
    total_rank = 32
    U = [[0.0] * total_rank for _ in range(n * m)]  # shape (4, 32)
    V = [[0.0] * total_rank for _ in range(m * p)]  # shape (18, 32)
    W = [[0.0] * total_rank for _ in range(p * n)]  # shape (18, 32)
    
    # Strassen linear forms for 2x2x2
    # u_r are linear forms in A (indices 0,1,2,3 corresponding to A[0,0], A[0,1], A[1,0], A[1,1])
    # v_r are linear forms in B_local (indices 0,1,2,3 corresponding to B[0,0], B[0,1], B[1,0], B[1,1])
    # w_r are linear forms in C_local (indices 0,1,2,3 corresponding to C[0,0], C[0,1], C[1,0], C[1,1])
    
    strassen_u = [
        [1, 0, 0, 1],  # a + d
        [0, 0, 1, 1],  # c + d
        [1, 0, 0, 0],  # a
        [0, 0, 0, 1],  # d
        [1, 1, 0, 0],  # a + b
        [1, 0, -1, 0], # c - a
        [0, 1, 0, -1]  # b - d
    ]
    
    strassen_v = [
        [1, 0, 0, 1],  # e + h
        [1, 0, 0, 0],  # e
        [0, 1, 0, -1], # f - h
        [0, -1, 1, 0], # g - e
        [0, 0, 0, 1],  # h
        [1, 1, 0, 0],  # e + f
        [0, 0, 1, 1]   # g + h
    ]
    
    strassen_w = [
        [1, 0, 0, 1],  # C11 + C22
        [0, 0, 1, 0],  # C21
        [0, 1, 0, 0],  # C12
        [0, 0, 1, 0],  # C21 (Wait, M4 is d(g-e) -> C21 and C22? No.)
        # Let's re-verify Strassen C formulas.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # M1: C11, C22
        # M2: C21, C22 (with -1)
        # M3: C12, C22
        # M4: C11, C21
        # M5: C11 (-1), C12
        # M6: C22
        # M7: C11
    ]
    
    # Correct w vectors:
    strassen_w = [
        [1, 0, 0, 1],  # M1: C11, C22
        [0, 0, 1, -1], # M2: C21, -C22
        [0, 1, 0, 1],  # M3: C12, C22
        [1, 0, 1, 0],  # M4: C11, C21
        [-1, 1, 0, 0], # M5: -C11, C12
        [0, 0, 0, 1],  # M6: C22
        [1, 0, 0, 0]   # M7: C11
    ]
    
    # Schoolbook for 2x2x1 (Rank 4)
    # C1 = A11 B1 + A12 B2
    # C2 = A21 B1 + A22 B2
    # Terms:
    # 1. A11 * B1 -> C1
    # 2. A12 * B2 -> C1
    # 3. A21 * B1 -> C2
    # 4. A22 * B2 -> C2
    
    schoolbook_u = [
        [1, 0, 0, 0], # A11
        [0, 1, 0, 0], # A12
        [0, 0, 1, 0], # A21
        [0, 0, 0, 1]  # A22
    ]
    # B_local for 2x2x1 has indices 0,1 (B1, B2)
    # But our V matrix has size 18.
    # We will map these to the specific column.
    
    def add_strassen_block(col_offset):
        """Adds Strassen decomposition for a 2x2x2 block at column offset col_offset."""
        for r in range(7):
            idx = col_offset * 4 + r # This is wrong, rank is 7 per block.
            # We need a global rank index.
            pass

    # Let's build the matrices directly.
    # Rank indices:
    # 0-6: Block 0
    # 7-13: Block 1
    # 14-20: Block 2
    # 21-27: Block 3
    # 28-31: Block 4
    
    for b in range(4):
        rank_offset = b * 7
        b_col_offset = b * 2
        c_col_offset = b * 2
        
        for r in range(7):
            idx = rank_offset + r
            
            # U
            for i in range(4):
                U[i][idx] = strassen_u[r][i]
                
            # V
            # Map local B indices (0,1,2,3) to global B indices
            # Local 0 -> B[0, b_col_offset] -> index 0*9 + b_col_offset
            # Local 1 -> B[0, b_col_offset+1] -> index 0*9 + b_col_offset+1
            # Local 2 -> B[1, b_col_offset] -> index 1*9 + b_col_offset
            # Local 3 -> B[1, b_col_offset+1] -> index 1*9 + b_col_offset+1
            
            local_to_global_b = [
                0 * 9 + b_col_offset,
                0 * 9 + b_col_offset + 1,
                1 * 9 + b_col_offset,
                1 * 9 + b_col_offset + 1
            ]
            
            for k in range(4):
                V[local_to_global_b[k]][idx] = strassen_v[r][k]
                
            # W
            # Map local C indices (0,1,2,3) to global C indices
            # Local 0 -> C[0, c_col_offset] -> index 0*2 + c_col_offset
            # Local 1 -> C[0, c_col_offset+1] -> index 0*2 + c_col_offset+1
            # Local 2 -> C[1, c_col_offset] -> index 1*2 + c_col_offset
            # Local 3 -> C[1, c_col_offset+1] -> index 1*2 + c_col_offset+1
            
            local_to_global_c = [
                0 * 2 + c_col_offset,
                0 * 2 + c_col_offset + 1,
                1 * 2 + c_col_offset,
                1 * 2 + c_col_offset + 1
            ]
            
            for k in range(4):
                W[local_to_global_c[k]][idx] = strassen_w[r][k]
                
    # Block 4: Column 8 (Rank 4)
    rank_offset = 28
    b_col_offset = 8
    c_col_offset = 8
    
    # Schoolbook terms
    # 1. A11 * B1 -> C1
    # 2. A12 * B2 -> C1
    # 3. A21 * B1 -> C2
    # 4. A22 * B2 -> C2
    
    # Term 1
    idx = rank_offset
    U[0][idx] = 1.0 # A11
    V[0 * 9 + b_col_offset][idx] = 1.0 # B1 (row 0, col 8)
    W[0 * 2 + c_col_offset][idx] = 1.0 # C1 (row 0, col 8)
    
    # Term 2
    idx = rank_offset + 1
    U[1][idx] = 1.0 # A12
    V[1 * 9 + b_col_offset][idx] = 1.0 # B2 (row 1, col 8)
    W[0 * 2 + c_col_offset][idx] = 1.0 # C1
    
    # Term 3
    idx = rank_offset + 2
    U[2][idx] = 1.0 # A21
    V[0 * 9 + b_col_offset][idx] = 1.0 # B1
    W[1 * 2 + c_col_offset][idx] = 1.0 # C2 (row 1, col 8)
    
    # Term 4
    idx = rank_offset + 3
    U[3][idx] = 1.0 # A22
    V[1 * 9 + b_col_offset][idx] = 1.0 # B2
    W[1 * 2 + c_col_offset][idx] = 1.0 # C2
    
    return {"U": U, "V": V, "W": W}
