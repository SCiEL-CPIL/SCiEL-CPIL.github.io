"""Matrix multiplication 4x4 by 4x8 using nested Strassen decomposition."""
DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # We'll build the decomposition using nested Strassen.
    # Top level: 2x2 block Strassen for the outer structure
    # Each block multiplication is 2x2 * 2x4, decomposed as two 2x2*2x2 Strassen
    
    # First, define the base Strassen factors for 2x2 * 2x2 -> 2x2
    # Strassen: C = A*B, rank 7
    # U factors: for each Strassen term, extract the needed A combination
    # V factors: for each Strassen term, extract the needed B combination  
    # W factors: combine to get C elements
    
    # Strassen 2x2:
    # M1 = (A22)*(B11-B21)
    # M2 = (A11+A21)*(B21)
    # M3 = (A11+A22)*(B12)
    # M4 = (A11)*(B12-B22)
    # M5 = (A12-A22)*(B11+B21)
    # M6 = (A11-A12)*(B22+B12)
    # M7 = (A21-A11)*(B11+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For 2x2*2x2, U has shape (4,7), V has shape (4,7), W has shape (4,7)
    # U[i,j] corresponds to coefficient of A element i in Strassen term j
    # A elements: index 0=A11, 1=A12, 2=A21, 3=A22 (row-major, 2x2)
    # B elements: index 0=B11, 1=B12, 2=B21, 3=B22
    # C elements: index 0=C11, 1=C12, 2=C21, 3=C22
    
    U_strassen_2 = [
        # M1: A22 only
        [0, 0, 0, 1],
        # M2: A11+A21
        [1, 0, 1, 0],
        # M3: A11+A22
        [1, 0, 0, 1],
        # M4: A11 only
        [1, 0, 0, 0],
        # M5: A12-A22
        [0, 1, 0, -1],
        # M6: A11-A12
        [1, -1, 0, 0],
        # M7: A21-A11
        [-1, 0, 1, 0],
    ]
    V_strassen_2 = [
        # M1: B11-B21
        [1, 0, -1, 0],
        # M2: B21
        [0, 0, 1, 0],
        # M3: B12
        [0, 1, 0, 0],
        # M4: B12-B22
        [0, 1, 0, -1],
        # M5: B11+B21
        [1, 0, 1, 0],
        # M6: B22+B12
        [0, 1, 0, 1],
        # M7: B11+B22
        [1, 0, 0, 1],
    ]
    W_strassen_2 = [
        # C11 = M1+M4-M5+M7
        [1, 0, 0, 1, -1, 0, 1],
        # C12 = M3+M5
        [0, 0, 1, 0, 1, 0, 0],
        # C21 = M2+M4
        [0, 1, 0, 1, 0, 0, 0],
        # C22 = M1-M2+M3+M6
        [1, -1, 1, 0, 0, 1, 0],
    ]
    
    # Now, the top-level structure for 4x4 * 4x8:
    # A is 4x4, split into 2x2 blocks: A = [[A_00, A_01], [A_10, A_11]]
    # B is 4x8, split into 2x4 blocks: B = [[B_00, B_01], [B_10, B_11]]
    # C is 4x8, split into 2x4 blocks: C = [[C_00, C_01], [C_10, C_11]]
    
    # Each block multiplication is 2x2 * 2x4 -> 2x4
    # We decompose 2x2 * 2x4 as: compute A*B_left and A*B_right separately
    # Each is 2x2 * 2x2 with rank 7, so total rank 14 per block mult
    
    # Top-level Strassen (7 terms), each term is a 2x2*2x4 mult (rank 14)
    # Total rank: 7 * 14 = 98
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * rank for _ in range(p * n)]  # 32 x 98
    
    # Top-level Strassen coefficients for the 2x2 block structure
    # A blocks: index 0=A_00, 1=A_01, 2=A_10, 3=A_11 (each 2x2)
    # B blocks: index 0=B_00, 1=B_01, 2=B_10, 3=B_11 (each 2x4)
    # C blocks: index 0=C_00, 1=C_01, 2=C_10, 3=C_11 (each 2x4)
    
    # Top-level Strassen:
    # T1 = A_11 * (B_00 - B_10)
    # T2 = (A_00 + A_10) * B_10
    # T3 = (A_00 + A_11) * B_01
    # T4 = A_00 * (B_01 - B_11)
    # T5 = (A_01 - A_11) * (B_00 + B_10)
    # T6 = (A_00 - A_01) * (B_11 + B_01)
    # T7 = (A_10 - A_00) * (B_00 + B_11)
    # C_00 = T1 + T4 - T5 + T7
    # C_01 = T3 + T5
    # C_10 = T2 + T4
    # C_11 = T1 - T2 + T3 + T6
    
    # A block coefficients (which A blocks appear in each term, with what sign)
    # For each term, list of (block_idx, coefficient)
    A_coeffs = [
        [(3, 1)],           # T1: A_11
        [(0, 1), (2, 1)],  # T2: A_00 + A_10
        [(0, 1), (3, 1)],  # T3: A_00 + A_11
        [(0, 1)],           # T4: A_00
        [(1, 1), (3, -1)], # T5: A_01 - A_11
        [(0, 1), (1, -1)], # T6: A_00 - A_01
        [(2, 1), (0, -1)], # T7: A_10 - A_00
    ]
    
    # B block coefficients
    B_coeffs = [
        [(0, 1), (2, -1)], # T1: B_00 - B_10
        [(2, 1)],           # T2: B_10
        [(1, 1)],           # T3: B_01
        [(1, 1), (3, -1)], # T4: B_01 - B_11
        [(0, 1), (2, 1)],  # T5: B_00 + B_10
        [(3, 1), (1, 1)],  # T6: B_11 + B_01
        [(0, 1), (3, 1)],  # T7: B_00 + B_11
    ]
    
    # C block coefficients
    C_coeffs = [
        [(0, 1), (3, 1), (4, -1), (6, 1)],  # C_00 = T1+T4-T5+T7
        [(2, 1), (4, 1)],                   # C_01 = T3+T5
        [(1, 1), (3, 1)],                   # C_10 = T2+T4
        [(0, 1), (1, -1), (2, 1), (5, 1)], # C_11 = T1-T2+T3+T6
    ]
    
    # Now build the full decomposition
    # For each top-level term t (0-6), and each sub-term s (0-6 for Strassen within 2x2*2x2),
    # and for each of the 2 right-half sub-multiplications (left B half, right B half)
    
    # Each 2x2*2x4 multiplication is split into two 2x2*2x2 multiplications
    # So for each top-level term, we have 2 (left/right B half) * 7 (Strassen) = 14 sub-terms
    # Total: 7 * 14 = 98
    
    col = 0
    for t in range(7):  # top-level Strassen term
        for half in range(2):  # left half (cols 0-3 of B) or right half (cols 4-7 of B)
            for s in range(7):  # bottom-level Strassen term
                # Build U factor
                # U rows: 16 (4x4 A matrix, row-major: i*4+j)
                # A block (bi) covers rows: bi//2 * 2 + {0,1} for row, bi%2 * 2 + {0,1} for col
                # Block 0 (A_00): rows 0,1 cols 0,1 -> U indices 0,1,4,5
                # Block 1 (A_01): rows 0,1 cols 2,3 -> U indices 2,3,6,7
                # Block 2 (A_10): rows 2,3 cols 0,1 -> U indices 8,9,12,13
                # Block 3 (A_11): rows 2,3 cols 2,3 -> U indices 10,11,14,15
                
                for (block_idx, coeff) in A_coeffs[t]:
                    # Map block_idx to the 4 positions in the 2x2 block
                    br = block_idx // 2  # block row (0 or 1)
                    bc = block_idx % 2   # block col (0 or 1)
                    for dr in range(2):
                        for dc in range(2):
                            u_idx = (br * 2 + dr) * 4 + (bc * 2 + dc)
                            # The bottom Strassen U factor
                            u_sub_idx = dr * 2 + dc  # 0,1,2,3 for the 2x2 block
                            U[u_idx][col] = coeff * U_strassen_2[s][u_sub_idx]
                
                # Build V factor
                # V rows: 32 (4x8 B matrix, row-major: i*8+j)
                # B block 0 (B_00): rows 0,1 cols 0,3 -> V indices 0-3, 8-11
                # B block 1 (B_01): rows 0,1 cols 4,7 -> V indices 4-7, 12-15
                # B block 2 (B_10): rows 2,3 cols 0,3 -> V indices 16-19, 24-27
                # B block 3 (B_11): rows 2,3 cols 4,7 -> V indices 20-23, 28-31
                
                for (block_idx, coeff) in B_coeffs[t]:
                    br = block_idx // 2  # block row
                    bc = block_idx % 2   # block col (0=left, 1=right)
                    # The 'half' parameter tells us which 2-column strip within the 2x4 block
                    # For B_00 (cols 0-3): half=0 -> cols 0-1, half=1 -> cols 2-3
                    # For B_01 (cols 4-7): half=0 -> cols 4-5, half=1 -> cols 6-7
                    # Actually, let me reconsider: B_00 is 2x4 with cols 0-3
                    # half=0: first 2 cols, half=1: last 2 cols
                    
                    base_col = br * 8 + bc * 4 + half * 2
                    for dr in range(2):
                        for dc in range(2):
                            v_idx = (br * 2 + dr) * 8 + (bc * 4 + half * 2 + dc)
                            v_sub_idx = dr * 2 + dc  # 0,1,2,3 for the 2x2 block
                            V[v_idx][col] = coeff * V_strassen_2[s][v_sub_idx]
                
                # Build W factor
                # W rows: 32 (8x4 C matrix, row-major: k*4+i per task: W[k*n+i, r])
                # W[k*4+i] where k is column of C (0-7), i is row of C (0-3)
                # C block 0 (C_00): rows 0,1 cols 0,3 -> W indices 0-1, 4-5, 16-17, 20-21
                # C block 1 (C_01): rows 0,1 cols 4,7 -> W indices 2-3, 6-7, 18-19, 22-23
                # C block 2 (C_10): rows 2,3 cols 0,3 -> W indices 8-9, 12-13, 24-25, 28-29
                # C block 3 (C_11): rows 2,3 cols 4,7 -> W indices 10-11, 14-15, 26-27, 30-31
                
                for (c_block_idx, coeff) in C_coeffs[t]:
                    cr = c_block_idx // 2  # C block row
                    cc = c_block_idx % 2   # C block col (0=left, 1=right)
                    base_k = cc * 4 + half * 2
                    for dr in range(2):
                        for dc in range(2):
                            # C row: cr*2+dr, C col: cc*4+half*2+dc
                            w_idx = (cc * 4 + half * 2 + dc) * 4 + (cr * 2 + dr)
                            w_sub_idx = dr * 2 + dc  # 0,1,2,3 for the 2x2 block
                            W[w_idx][col] = coeff * W_strassen_2[s][w_sub_idx]
                
                col += 1
    
    return {"U": U, "V": V, "W": W}
