"""Program for matrix multiplication 4,4,9 tensor decomposition."""
DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9
    
    # We'll use a block decomposition approach.
    # Split the 9 columns of B into two 4-column blocks and one 1-column block.
    # Block 1: 4x4x4 using Strassen (Rank 49)
    # Block 2: 4x4x4 using Strassen (Rank 49)  
    # Block 3: 4x4x1 using Schoolbook (Rank 16)
    # Total Rank: 49 + 49 + 16 = 114
    
    def get_strassen_2x2():
        """Returns U, V, W for 2x2x2 Strassen decomposition."""
        # U: 4x7, V: 4x7, W: 4x7
        # Indices: A: 0,1,2,3 (row-major), B: 0,1,2,3, C: 0,1,2,3
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1 = (A00+A11)(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1
        
        # M3 = A00(B10-B11)
        U[0][2] = 1
        V[2][2] = 1; V[3][2] = -1
        W[1][2] = 1; W[3][2] = 1
        
        # M4 = A11(B01-B00)
        U[3][3] = 1
        V[1][3] = 1; V[0][3] = -1
        W[1][3] = 1; W[3][3] = 1
        
        # M5 = (A00+A01)B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        W[3][5] = 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[1][6] = -1
        
        return U, V, W
    
    def get_strassen_4x4():
        """Returns U, V, W for 4x4x4 Strassen decomposition (rank 49)."""
        # We build this by applying Strassen 2x2 to the 7 linear combinations
        # for 4x4 Strassen, each of which is a 2x2x2 multiplication.
        
        U = [[0]*49 for _ in range(16)]
        V = [[0]*49 for _ in range(16)]
        W = [[0]*49 for _ in range(16)]
        
        u2, v2, w2 = get_strassen_2x2()
        
        # 4x4 Strassen linear combinations:
        # A = [A11 A12; A21 A22], B = [B11 B12; B21 B22]
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # Each Mk is computed as a 2x2x2 Strassen product.
        # We need to define for each Mk:
        # - Which 2x2 blocks of A are used with what coefficients
        # - Which 2x2 blocks of B are used with what coefficients
        # - Which 2x2 blocks of C receive the result with what coefficients
        
        # Block indices in 4x4 (row-major):
        # A11: rows 0-1, cols 0-1 -> indices 0,1,4,5
        # A12: rows 0-1, cols 2-3 -> indices 2,3,6,7
        # A21: rows 2-3, cols 0-1 -> indices 8,9,12,13
        # A22: rows 2-3, cols 2-3 -> indices 10,11,14,15
        
        def apply_strassen_2x2(a_blocks, b_blocks, c_blocks, rank_offset):
            """
            Apply Strassen 2x2 with given block coefficients.
            a_blocks: list of (block_idx, coeff) for A
            b_blocks: list of (block_idx, coeff) for B
            c_blocks: list of (block_idx, coeff) for C
            """
            for k in range(7):
                # For each of the 7 Strassen terms, apply the local 2x2 structure
                for local_i in range(4):
                    # local_i maps to (row, col) in 2x2: (local_i//2, local_i%2)
                    # The coefficient for this element in the linear combo
                    a_coeff = 0
                    for block_idx, coeff in a_blocks:
                        # Check if local_i corresponds to an element in this block
                        # Block 0 (A11): elements 0,1,4,5 -> local 0,1,2,3
                        # Block 1 (A12): elements 2,3,6,7 -> local 0,1,2,3
                        # Block 2 (A21): elements 8,9,12,13 -> local 0,1,2,3
                        # Block 3 (A22): elements 10,11,14,15 -> local 0,1,2,3
                        if block_idx == 0:
                            global_idx = [0,1,4,5][local_i]
                        elif block_idx == 1:
                            global_idx = [2,3,6,7][local_i]
                        elif block_idx == 2:
                            global_idx = [8,9,12,13][local_i]
                        elif block_idx == 3:
                            global_idx = [10,11,14,15][local_i]
                        else:
                            continue
                        a_coeff += coeff * u2[local_i][k]
                    
                    if a_coeff != 0:
                        # Determine global index
                        if block_idx == 0:
                            g_idx = [0,1,4,5][local_i]
                        elif block_idx == 1:
                            g_idx = [2,3,6,7][local_i]
                        elif block_idx == 2:
                            g_idx = [8,9,12,13][local_i]
                        elif block_idx == 3:
                            g_idx = [10,11,14,15][local_i]
                        else:
                            continue
                        U[g_idx][rank_offset + k] += a_coeff
                    
                    b_coeff = 0
                    for block_idx, coeff in b_blocks:
                        if block_idx == 0:
                            global_idx = [0,1,4,5][local_i]
                        elif block_idx == 1:
                            global_idx = [2,3,6,7][local_i]
                        elif block_idx == 2:
                            global_idx = [8,9,12,13][local_i]
                        elif block_idx == 3:
                            global_idx = [10,11,14,15][local_i]
                        else:
                            continue
                        b_coeff += coeff * v2[local_i][k]
                    
                    if b_coeff != 0:
                        if block_idx == 0:
                            g_idx = [0,1,4,5][local_i]
                        elif block_idx == 1:
                            g_idx = [2,3,6,7][local_i]
                        elif block_idx == 2:
                            g_idx = [8,9,12,13][local_i]
                        elif block_idx == 3:
                            g_idx = [10,11,14,15][local_i]
                        else:
                            continue
                        V[g_idx][rank_offset + k] += b_coeff
                    
                    c_coeff = 0
                    for block_idx, coeff in c_blocks:
                        if block_idx == 0:
                            global_idx = [0,1,4,5][local_i]
                        elif block_idx == 1:
                            global_idx = [2,3,6,7][local_i]
                        elif block_idx == 2:
                            global_idx = [8,9,12,13][local_i]
                        elif block_idx == 3:
                            global_idx = [10,11,14,15][local_i]
                        else:
                            continue
                        c_coeff += coeff * w2[local_i][k]
                    
                    if c_coeff != 0:
                        if block_idx == 0:
                            g_idx = [0,1,4,5][local_i]
                        elif block_idx == 1:
                            g_idx = [2,3,6,7][local_i]
                        elif block_idx == 2:
                            g_idx = [8,9,12,13][local_i]
                        elif block_idx == 3:
                            g_idx = [10,11,14,15][local_i]
                        else:
                            continue
                        W[g_idx][rank_offset + k] += c_coeff
        
        # M1 = (A11+A22)(B11+B22)
        # C11 += M1, C22 += M1
        apply_strassen_2x2([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)], 0)
        
        # M2 = (A21+A22)B11
        # C21 += M2, C22 += M2
        apply_strassen_2x2([(2,1),(3,1)], [(0,1)], [(2,1),(3,1)], 7)
        
        # M3 = A11(B21-B22)
        # C11 -= M3, C12 += M3
        apply_strassen_2x2([(0,1)], [(2,1),(3,-1)], [(0,-1),(1,1)], 14)
        
        # M4 = A22(B12-B11)
        # C21 += M4, C22 -= M4
        apply_strassen_2x2([(3,1)], [(1,1),(0,-1)], [(2,1),(3,-1)], 21)
        
        # M5 = (A11+A12)B22
        # C12 += M5, C22 += M5
        apply_strassen_2x2([(0,1),(1,1)], [(3,1)], [(1,1),(3,1)], 28)
        
        # M6 = (A21-A11)(B11+B12)
        # C11 += M6, C21 += M6
        apply_strassen_2x2([(2,1),(0,-1)], [(0,1),(1,1)], [(0,1),(2,1)], 35)
        
        # M7 = (A12-A22)(B21+B22)
        # C11 -= M7, C12 += M7
        apply_strassen_2x2([(1,1),(3,-1)], [(2,1),(3,1)], [(0,-1),(1,1)], 42)
        
        return U, V, W
    
    def get_schoolbook_4x4_1():
        """Returns U, V, W for 4x4x1 schoolbook decomposition."""
        # U: 16x16, V: 4x16, W: 4x16
        U = [[0]*16 for _ in range(16)]
        V = [[0]*16 for _ in range(4)]
        W = [[0]*16 for _ in range(4)]
        
        col = 0
        for i in range(4):
            for j in range(4):
                U[i*4+j][col] = 1
                V[j][col] = 1
                W[i][col] = 1
                col += 1
        return U, V, W
    
    # Assemble the full 4x4x9 decomposition.
    # Split B into two 4-column blocks and one 1-column block.
    # Block 1: 4x4x4 (Strassen, rank 49)
    # Block 2: 4x4x4 (Strassen, rank 49)  
    # Block 3: 4x4x1 (Schoolbook, rank 16)
    # Total rank: 114
    
    total_rank = 49 + 49 + 16
    U_full = [[0]*total_rank for _ in range(16)]
    V_full = [[0]*total_rank for _ in range(36)]
    W_full = [[0]*total_rank for _ in range(36)]
    
    # Helper to add a block decomposition
    def add_block(u_block, v_block, w_block, v_offset, w_offset, rank_offset):
        # u_block: 16 x r
        # v_block: 4 x r
        # w_block: 4 x r
        for i in range(16):
            for k in range(len(u_block[0])):
                if u_block[i][k] != 0:
                    U_full[i][rank_offset + k] += u_block[i][k]
        
        for i in range(4):
            for k in range(len(v_block[0])):
                if v_block[i][k] != 0:
                    V_full[v_offset + i][rank_offset + k] += v_block[i][k]
        
        for i in range(4):
            for k in range(len(w_block[0])):
                if w_block[i][k] != 0:
                    W_full[w_offset + i][rank_offset + k] += w_block[i][k]
    
    # Block 1: 4x4x4 Strassen (columns 0-3 of B)
    u1, v1, w1 = get_strassen_4x4()
    add_block(u1, v1, w1, 0, 0, 0)
    
    # Block 2: 4x4x4 Strassen (columns 4-7 of B)
    u2, v2, w2 = get_strassen_4x4()
    add_block(u2, v2, w2, 4, 4, 49)
    
    # Block 3: 4x4x1 Schoolbook (column 8 of B)
    u3, v3, w3 = get_schoolbook_4x4_1()
    add_block(u3, v3, w3, 8, 8, 98)
    
    return {"U": U_full, "V": V_full, "W": W_full}
