import time
import sys

DIMENSIONS = [4, 4, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    
    # --- Part 1: 4x4x4 Strassen Decomposition (Rank 49) ---
    
    # Define 2x2 Strassen decomposition
    # U2: 4x7, V2: 4x7, W2: 4x7
    # Rows of U2/V2/W2 correspond to indices 0..3 of 2x2 matrices
    # Cols correspond to the 7 multiplications
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # M1 = (a00+a11)(b00+b11) -> c00, c11
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # M2 = (a10+a11)b00 -> c10, c11
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    
    # M3 = a00(b10-b11) -> c00, c01
    U2[0][2] = 1
    V2[2][2] = 1; V2[3][2] = -1
    W2[0][2] = 1; W2[1][2] = 1
    
    # M4 = a11(b01-b00) -> c00, c11
    U2[3][3] = 1
    V2[1][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[3][3] = 1
    
    # M5 = (a00+a01)b11 -> c00, c01
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # M6 = (a10-a00)(b00+b10) -> c11
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[2][5] = 1
    W2[3][5] = 1
    
    # M7 = (a01-a11)(b11+b01) -> c00, c11
    U2[1][6] = 1; U2[3][6] = -1
    V2[3][6] = 1; V2[1][6] = 1
    W2[0][6] = 1; W2[3][6] = -1

    # Compose 4x4 Strassen
    # U4: 16x49, V4: 16x49, W4: 16x49
    U4 = [[0]*49 for _ in range(16)]
    V4 = [[0]*49 for _ in range(16)]
    W4 = [[0]*49 for _ in range(16)]
    
    # Outer Strassen terms definitions
    # A_blocks: 00, 01, 10, 11
    # Each block is 2x2. Indices in 4x4 matrix:
    # A00: 0,1,4,5
    # A01: 2,3,6,7
    # A10: 8,9,12,13
    # A11: 10,11,14,15
    
    # B_blocks similar
    
    # We define the linear combinations for A and B for each of the 7 outer terms
    # term: (A_comb, B_comb)
    # A_comb is a dict: {block_idx: coeff}
    # block_idx: 0 (00), 1 (01), 2 (10), 3 (11)
    
    outer_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}),       # M2
        ({0: 1}, {2: 1, 3: -1}),      # M3
        ({3: 1}, {1: 1, 0: -1}),      # M4
        ({0: 1, 1: 1}, {3: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 2: 1}),# M6
        ({1: 1, 3: -1}, {3: 1, 1: 1}) # M7
    ]
    
    # Mapping from block index to 4x4 row/col indices
    # Block 0 (00): rows 0-1, cols 0-1 -> indices 0,1,4,5
    # Block 1 (01): rows 0-1, cols 2-3 -> indices 2,3,6,7
    # Block 2 (10): rows 2-3, cols 0-1 -> indices 8,9,12,13
    # Block 3 (11): rows 2-3, cols 2-3 -> indices 10,11,14,15
    
    block_indices = {
        0: [0, 1, 4, 5],
        1: [2, 3, 6, 7],
        2: [8, 9, 12, 13],
        3: [10, 11, 14, 15]
    }
    
    # Inner Strassen indices
    # U2 rows 0..3 correspond to a00, a01, a10, a11 of the 2x2 block
    # So row 0 of U2 corresponds to index 0 of the block (top-left)
    # Row 1 of U2 corresponds to index 1 of the block (top-right)
    # etc.
    
    for t_idx, (a_comb, b_comb) in enumerate(outer_terms):
        # For each of the 7 inner terms
        for k_idx in range(7):
            col_idx = t_idx * 7 + k_idx
            
            # Process U4
            # U2 is 4x7. We are looking at column k_idx of U2.
            # U2[row][k_idx] is the coefficient for the row-th entry of the 2x2 A-block.
            # The row-th entry of the 2x2 A-block corresponds to a specific index in the 4x4 A matrix.
            # The 2x2 A-block is formed by linear combination of A's blocks.
            # Let A_block be the 2x2 matrix. Its entries are A_block[r][c].
            # A_block[r][c] = sum_{b} coeff_b * A_block_b[r][c]
            # So the coefficient for A[global_idx] is sum_{b} coeff_b * U2[local_idx][k_idx]
            # where local_idx is the index of the entry in the 2x2 block.
            
            # Actually, it's simpler:
            # The input to the inner Strassen is the matrix A_t.
            # A_t is a 2x2 matrix.
            # The entries of A_t are linear combinations of A's entries.
            # U2 defines how A_t's entries are combined to form the 7 products.
            # So, for each entry of A_t (say, top-left), its value is sum_{b} coeff_b * A_block_b[0][0].
            # The contribution to U4 is U2[0][k_idx] * (sum of coeff_b for the block containing A[global_idx]).
            
            # Let's iterate over the 4 entries of the 2x2 block (0,1,2,3)
            for local_r in range(4):
                coeff_u2 = U2[local_r][k_idx]
                if coeff_u2 == 0: continue
                
                # local_r corresponds to which entry of the 2x2 block?
                # U2 rows: 0:a00, 1:a01, 2:a10, 3:a11
                # So local_r=0 is top-left, 1 is top-right, 2 is bottom-left, 3 is bottom-right.
                
                # Which global indices in A contribute to this entry of A_t?
                # A_t[local_r] = sum_{b in a_comb} coeff_b * A_block_b[local_r]
                # A_block_b[local_r] is a single element of the 4x4 matrix A.
                
                for b_idx, coeff_a in a_comb.items():
                    # Get the global index of the local_r-th element of block b_idx
                    # Block b_idx is 2x2.
                    # local_r=0 -> row 0, col 0 of block
                    # local_r=1 -> row 0, col 1 of block
                    # local_r=2 -> row 1, col 0 of block
                    # local_r=3 -> row 1, col 1 of block
                    
                    row_in_block = local_r // 2
                    col_in_block = local_r % 2
                    
                    # Get global index
                    # Block b_idx has a base row and col
                    # b_idx 0: row 0, col 0
                    # b_idx 1: row 0, col 2
                    # b_idx 2: row 2, col 0
                    # b_idx 3: row 2, col 2
                    
                    base_row = (b_idx // 2) * 2
                    base_col = (b_idx % 2) * 2
                    
                    global_idx = (base_row + row_in_block) * 4 + (base_col + col_in_block)
                    
                    U4[global_idx][col_idx] += coeff_a * coeff_u2
            
            # Process V4 (similar logic)
            for local_c in range(4):
                coeff_v2 = V2[local_c][k_idx]
                if coeff_v2 == 0: continue
                
                for b_idx, coeff_b in b_comb.items():
                    row_in_block = local_c // 2
                    col_in_block = local_c % 2
                    
                    base_row = (b_idx // 2) * 2
                    base_col = (b_idx % 2) * 2
                    
                    global_idx = (base_row + row_in_block) * 4 + (base_col + col_in_block)
                    
                    V4[global_idx][col_idx] += coeff_b * coeff_v2
            
            # Process W4
            # W2 rows: 0:c00, 1:c01, 2:c10, 3:c11
            # These correspond to the output blocks of the 4x4 matrix C.
            # The outer Strassen combines these blocks to form the final 4x4 C.
            # Wait, the outer Strassen result is C = [[C00, C01], [C10, C11]]
            # where Cij are 2x2 blocks.
            # The inner Strassen produces the 2x2 block C_t (which is one of C00, C01, etc?)
            # No, the outer Strassen terms M1..M7 are 2x2 matrices.
            # The final C is a linear combination of M1..M7.
            # M_t is a 2x2 matrix.
            # The inner Strassen decomposes M_t = A_t B_t.
            # So W2 gives the coefficients for the entries of M_t.
            # The entries of M_t contribute to specific blocks of C.
            
            # Which block of C does M_t contribute to?
            # From Strassen formulas:
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # So for a given outer term t_idx (0..6 corresponding to M1..M7),
            # we need to know which C-blocks it contributes to, and with what sign.
            
            # Let's map t_idx to contributions
            # t_idx 0 (M1): C00 (+1), C11 (+1)
            # t_idx 1 (M2): C10 (+1), C11 (-1)
            # t_idx 2 (M3): C00 (+1), C01 (+1)
            # t_idx 3 (M4): C00 (+1), C11 (+1)
            # t_idx 4 (M5): C00 (-1), C01 (+1)
            # t_idx 5 (M6): C11 (+1)
            # t_idx 6 (M7): C00 (+1), C11 (-1)
            
            contributions = {
                0: {0: 1, 3: 1},
                1: {2: 1, 3: -1},
                2: {0: 1, 1: 1},
                3: {0: 1, 3: 1},
                4: {0: -1, 1: 1},
                5: {3: 1},
                6: {0: 1, 3: -1}
            }
            
            c_contribs = contributions[t_idx]
            
            for local_w in range(4):
                coeff_w2 = W2[local_w][k_idx]
                if coeff_w2 == 0: continue
                
                # local_w corresponds to entry in the 2x2 result block M_t
                # local_w=0 (00), 1 (01), 2 (10), 3 (11)
                
                # Which C-blocks does this entry of M_t contribute to?
                # The entry M_t[local_w] is added to C_block[local_w] for each block in c_contribs.
                
                for c_block_idx, coeff_c in c_contribs.items():
                    # c_block_idx is 0..3 corresponding to C00, C01, C10, C11
                    # local_w is the position within the 2x2 block
                    
                    row_in_block = local_w // 2
                    col_in_block = local_w % 2
                    
                    base_row = (c_block_idx // 2) * 2
                    base_col = (c_block_idx % 2) * 2
                    
                    global_idx = (base_row + row_in_block) * 4 + (base_col + col_in_block)
                    
                    W4[global_idx][col_idx] += coeff_c * coeff_w2

    # --- Part 2: 4x4x1 Schoolbook Decomposition (Rank 16) ---
    
    U3 = [[0]*16 for _ in range(16)]
    V3 = [[0]*16 for _ in range(4)]
    W3 = [[0]*16 for _ in range(4)]
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(1):
                U3[i*4 + j][col] = 1
                V3[j*1 + k][col] = 1
                W3[k*4 + i][col] = 1
                col += 1

    # --- Assemble Final 4x4x9 Decomposition ---
    
    rank = 49 + 49 + 16
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(36)]
    W = [[0]*rank for _ in range(36)]
    
    # Copy U4 to first 49 cols
    for r in range(16):
        for c in range(49):
            U[r][c] = U4[r][c]
            
    # Copy U4 to next 49 cols
    for r in range(16):
        for c in range(49):
            U[r][49 + c] = U4[r][c]
            
    # Copy U3 to last 16 cols
    for r in range(16):
        for c in range(16):
            U[r][98 + c] = U3[r][c]
            
    # Copy V4 to V
    # V4 is 16x49. V is 36x114.
    # V4 rows correspond to j*4+k (j in 0..3, k in 0..3).
    # V rows correspond to j*9+k (j in 0..3, k in 0..8).
    
    for r4 in range(16):
        j = r4 // 4
        k = r4 % 4
        rV = j * 9 + k
        for c in range(49):
            V[rV][c] = V4[r4][c]
            V[rV][49 + c] = V4[r4][c] # Same V4 for both 4x4 parts
            
    # Copy V3 to V
    # V3 is 4x16. V3 rows correspond to j (j in 0..3).
    # V rows for k=8 are j*9 + 8.
    
    for r3 in range(4):
        j = r3
        rV = j * 9 + 8
        for c in range(16):
            V[rV][98 + c] = V3[r3][c]
            
    # Copy W4 to W
    # W4 is 16x49. W4 rows correspond to k*4+i (k in 0..3, i in 0..3).
    # W rows correspond to k*4+i (k in 0..8, i in 0..3).
    
    for r4 in range(16):
        k = r4 // 4
        i = r4 % 4
        rW = k * 4 + i
        for c in range(49):
            W[rW][c] = W4[r4][c]
            
        # For the second 4x4 part (cols 4-7 of C), k is 4..7
        # W4 row r4 corresponds to k' in 0..3.
        # So global k = k' + 4.
        rW2 = (k + 4) * 4 + i
        for c in range(49):
            W[rW2][49 + c] = W4[r4][c]
            
    # Copy W3 to W
    # W3 is 4x16. W3 rows correspond to i (i in 0..3).
    # W rows for k=8 are 8*4 + i.
    
    for r3 in range(4):
        i = r3
        rW = 8 * 4 + i
        for c in range(16):
            W[rW][98 + c] = W3[r3][c]
            
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
