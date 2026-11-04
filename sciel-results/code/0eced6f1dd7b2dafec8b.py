import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # We will construct the rank 98 decomposition.
    # Rank 49 for the first 4 columns of B, and 49 for the last 4 columns.
    
    # First, generate the 2x2 Strassen decomposition matrices
    # U22: 4x7, V22: 4x7, W22: 4x7
    U22 = [[0]*7 for _ in range(4)]
    V22 = [[0]*7 for _ in range(4)]
    W22 = [[0]*7 for _ in range(4)]
    
    # Strassen terms for 2x2
    # Term 1: (A00+A11)(B00+B11)
    U22[0][0] = 1; U22[3][0] = 1
    V22[0][0] = 1; V22[3][0] = 1
    W22[0][0] = 1; W22[3][0] = 1 # C00, C11
    
    # Term 2: (A10+A11)B00
    U22[2][1] = 1; U22[3][1] = 1
    V22[0][1] = 1
    W22[2][1] = 1; W22[3][1] = -1 # C10, C11
    
    # Term 3: A00(B01-B11)
    U22[0][2] = 1
    V22[1][2] = 1; V22[3][2] = -1
    W22[1][2] = 1; W22[3][2] = 1 # C01, C11
    
    # Term 4: A11(B10-B00)
    U22[3][3] = 1
    V22[2][3] = 1; V22[0][3] = -1
    W22[0][3] = 1; W22[2][3] = 1 # C00, C10
    
    # Term 5: (A00+A01)B11
    U22[0][4] = 1; U22[1][4] = 1
    V22[3][4] = 1
    W22[0][4] = -1; W22[1][4] = 1 # C00, C01
    
    # Term 6: (A10-A00)(B00+B01)
    U22[2][5] = 1; U22[0][5] = -1
    V22[0][5] = 1; V22[1][5] = 1
    W22[3][5] = 1 # C11
    
    # Term 7: (A01-A11)(B10+B11)
    U22[1][6] = 1; U22[3][6] = -1
    V22[2][6] = 1; V22[3][6] = 1
    W22[0][6] = 1 # C00

    # Now construct the 4x4x4 decomposition (rank 49)
    # This will be used twice, once for B[:, 0:4] and once for B[:, 4:8]
    
    # U44: 16x49
    # V44: 16x49 (for 4x4 B part)
    # W44: 16x49
    
    U44 = [[0]*49 for _ in range(16)]
    V44 = [[0]*49 for _ in range(16)]
    W44 = [[0]*49 for _ in range(16)]
    
    # Outer Strassen terms definitions
    # Each term t has:
    # - A-part: linear combo of A blocks
    # - B-part: linear combo of B blocks
    # - C-part: which C blocks it contributes to and with what weight
    
    # A blocks: 00, 01, 10, 11
    # Global A indices:
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    # B blocks: 00, 01, 10, 11
    # Global B indices (for 4x4 part):
    # B00: 0,1,4,5
    # B01: 2,3,6,7
    # B10: 8,9,12,13
    # B11: 10,11,14,15
    
    # C blocks: 00, 01, 10, 11
    # Global C indices:
    # C00: 0,1,4,5
    # C01: 2,3,6,7
    # C10: 8,9,12,13
    # C11: 10,11,14,15

    # Define the 7 outer terms
    # t: (A_blocks, B_blocks, C_contributions)
    # A_blocks: list of (block_idx, weight)
    # B_blocks: list of (block_idx, weight)
    # C_contributions: list of (block_idx, weight)
    
    terms = [
        # 1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # 2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # 3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # 4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # 5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # 6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # 7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    ]
    
    # Block to global index mapping
    # Block 0 (00): rows 0-1, cols 0-1
    # Block 1 (01): rows 0-1, cols 2-3
    # Block 2 (10): rows 2-3, cols 0-1
    # Block 3 (11): rows 2-3, cols 2-3
    
    def block_indices(block_idx):
        r_start = (block_idx // 2) * 2
        c_start = (block_idx % 2) * 2
        indices = []
        for dr in range(2):
            for dc in range(2):
                # Local index in 2x2 block: dr*2 + dc
                # Global index: (r_start+dr)*4 + (c_start+dc)
                global_idx = (r_start + dr) * 4 + (c_start + dc)
                local_idx = dr * 2 + dc
                indices.append((global_idx, local_idx))
        return indices

    # Precompute block mappings
    block_map = [block_indices(b) for b in range(4)]

    for t_idx in range(7):
        a_parts, b_parts, c_parts = terms[t_idx]
        
        # For each inner Strassen term s
        for s_idx in range(7):
            col = t_idx * 7 + s_idx
            
            # Update U
            # U22[la, s_idx] is the coefficient for local A index la
            for la in range(4):
                coeff_a_local = U22[la][s_idx]
                if coeff_a_local != 0:
                    # Find which global A indices contribute to this local index in the combined A matrix
                    # The combined A matrix is sum of weighted blocks
                    # X = sum w_b * A_block_b
                    # X[la] = sum w_b * A_block_b[la]
                    # So for each block b with weight w_b, we add w_b * coeff_a_local to global indices of block b at local index la
                    
                    for b_idx, w_b in a_parts:
                        # Find global index corresponding to local index la in block b_idx
                        # block_map[b_idx] is list of (global_idx, local_idx)
                        for g_idx, l_idx in block_map[b_idx]:
                            if l_idx == la:
                                U44[g_idx][col] += w_b * coeff_a_local
                                break

            # Update V
            for lb in range(4):
                coeff_b_local = V22[lb][s_idx]
                if coeff_b_local != 0:
                    for b_idx, w_b in b_parts:
                        for g_idx, l_idx in block_map[b_idx]:
                            if l_idx == lb:
                                V44[g_idx][col] += w_b * coeff_b_local
                                break

            # Update W
            # W22[lc, s_idx] is the coefficient for local C index lc
            # The local C index lc corresponds to an element in the 2x2 result matrix P_t
            # P_t contributes to C blocks with weights
            for lc in range(4):
                coeff_c_local = W22[lc][s_idx]
                if coeff_c_local != 0:
                    for c_idx, w_c in c_parts:
                        # Map local C index lc to global C index in block c_idx
                        # Local lc is dr*2+dc
                        # Global index in block c_idx:
                        r_start = (c_idx // 2) * 2
                        c_start = (c_idx % 2) * 2
                        dr = lc // 2
                        dc = lc % 2
                        global_i = r_start + dr
                        global_k = c_start + dc
                        global_w_idx = global_k * 4 + global_i
                        W44[global_w_idx][col] += w_c * coeff_c_local

    # Now we have U44, V44, W44 for 4x4x4.
    # We need to combine two such decompositions for 4x4x8.
    
    rank = 98
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]
    
    # First half: B columns 0-3
    for i in range(16):
        for c in range(49):
            U[i][c] = U44[i][c]
            V[i][c] = V44[i][c] # V indices for first 4 cols of B are 0-15
            W[i][c] = W44[i][c]
            
    # Second half: B columns 4-7
    # V indices for second 4 cols of B are 16-31
    # B_block_00 in second half corresponds to global B indices 16-19?
    # B is 4x8.
    # B[0:4, 0:4] is first half. Indices 0-15.
    # B[0:4, 4:8] is second half.
    # B[0,4] is index 4? No, B is m x p = 4 x 8.
    # V index is j*p + k.
    # j in 0..3, k in 0..7.
    # First half k=0..3.
    # Second half k=4..7.
    
    # The decomposition V44 was for a 4x4 B matrix.
    # Its indices 0-15 correspond to j*p+k with p=4.
    # But here p=8.
    # So for the first half, V indices are j*8 + k (k=0..3).
    # V44 indices were j*4 + k.
    # We need to map V44 indices to global V indices.
    
    # Let's re-evaluate V mapping.
    # In V44, row index is j*4+k.
    # In global V, row index is j*8+k.
    # For first half (k=0..3), global idx = j*8+k.
    # V44 idx = j*4+k.
    
    # Let's redo the filling of U, V, W with proper index mapping.
    
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(32)]
    W = [[0]*rank for _ in range(32)]
    
    # Fill first 49 columns
    for i in range(16):
        U[i][0:49] = U44[i]
        
    for v_idx_44 in range(16):
        j = v_idx_44 // 4
        k = v_idx_44 % 4
        v_idx_global = j * 8 + k
        V[v_idx_global][0:49] = V44[v_idx_44]
        
    for w_idx_44 in range(16):
        # W44 index was k*4+i for 4x4 C.
        # Global W index is k*4+i for 4x8 C?
        # C is n x p = 4 x 8.
        # W index is k*n + i = k*4 + i.
        # In 4x4 case, k was 0..3.
        # In 4x8 case, k is 0..7.
        # First half k=0..3.
        # So W44 index k*4+i is the same as global W index for k=0..3.
        W[w_idx_44][0:49] = W44[w_idx_44]
        
    # Fill second 49 columns
    # U is the same
    for i in range(16):
        U[i][49:98] = U44[i]
        
    # V for second half
    for v_idx_44 in range(16):
        j = v_idx_44 // 4
        k = v_idx_44 % 4
        # k in second half is 4..7
        k_global = k + 4
        v_idx_global = j * 8 + k_global
        V[v_idx_global][49:98] = V44[v_idx_44]
        
    # W for second half
    # W44 index k*4+i.
    # Here k corresponds to 4..7 in global C.
    # So global W index is (k+4)*4 + i.
    for w_idx_44 in range(16):
        k = w_idx_44 // 4
        i = w_idx_44 % 4
        w_idx_global = (k + 4) * 4 + i
        W[w_idx_global][49:98] = W44[w_idx_44]
        
    return {"U": U, "V": V, "W": W}
