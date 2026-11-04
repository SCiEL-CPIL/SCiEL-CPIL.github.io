"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Decompose M_{3,4,8} into 8 blocks of M_{3,2,2}.
    # For each M_{3,2,2} block, use Strassen (rank 7) for top 2x2 and Schoolbook (rank 4) for bottom 1x2.
    # Total rank = 8 * (7 + 4) = 88.
    n, m, p = DIMENSIONS  # 3, 4, 8
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen for 2x2: C = A*B where A is 2x2, B is 2x2
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For M_{3,2,2}: A is 3x2, B is 2x2
    # Split A into A_top (2x2) and A_bot (1x2)
    # C_top = A_top * B (2x2) -> use Strassen (7 terms)
    # C_bot = A_bot * B (1x2) -> use Schoolbook (2 terms)
    # Wait, A_bot is 1x2, B is 2x2, so C_bot is 1x2.
    # C_bot[0] = A_bot[0,0]*B[0,0] + A_bot[0,1]*B[1,0]
    # C_bot[1] = A_bot[0,0]*B[0,1] + A_bot[0,1]*B[1,1]
    # This is 4 terms in total for the 1x2 output? No, 2 outputs, each sum of 2 products.
    # In tensor decomposition, each product is a rank-1 term.
    # Term 1: A_bot[0,0] * B[0,0] * C_bot[0]
    # Term 2: A_bot[0,1] * B[1,0] * C_bot[0]
    # Term 3: A_bot[0,0] * B[0,1] * C_bot[1]
    # Term 4: A_bot[0,1] * B[1,1] * C_bot[1]
    # So 4 terms for the bottom part.
    # Total 7 + 4 = 11 terms per block.
    
    # Block structure:
    # A is 3x4. Split into 4 blocks of 3x2? Or 2 blocks of 3x4?
    # B is 4x8. Split into 8 blocks of 4x1? No.
    # Let's split B into 8 blocks of 4x1? No, B is 4x8.
    # Let's split the multiplication into 8 independent 3x2 * 2x2?
    # A is 3x4. Let A = [A1 A2] where A1, A2 are 3x2.
    # B is 4x8. Let B = [B1; B2] where B1, B2 are 2x8.
    # C = A1*B1 + A2*B2.
    # This is 2 blocks of 3x2 * 2x8.
    # Each 3x2 * 2x8 can be split into 4 blocks of 3x2 * 2x2.
    # So total 8 blocks of 3x2 * 2x2.
    
    # Let's define the 8 blocks.
    # Block b (0..7) corresponds to:
    #   A_block: rows 0..2, cols (b//4)*2 .. (b//4)*2+1
    #   B_block: rows (b%4)*2 .. (b%4)*2+1, cols (b//4)*2 .. (b//4)*2+1?
    # Wait, B is 4x8.
    # If we split A into 2 blocks of 3x2 (cols 0-1, 2-3).
    # And B into 2 blocks of 2x8 (rows 0-1, 2-3).
    # Then C = A1*B1 + A2*B2.
    # A1*B1: 3x2 * 2x8 -> 3x8.
    # A2*B2: 3x2 * 2x8 -> 3x8.
    # Split A1*B1 into 4 blocks of 3x2 * 2x2.
    #   A1 is 3x2. B1 is 2x8. Split B1 into 4 blocks of 2x2.
    #   So 4 blocks for A1*B1.
    # Similarly 4 blocks for A2*B2.
    # Total 8 blocks.
    
    # Block indices:
    # b = 0..3: A1 * B1[:, 2*(b%4) : 2*(b%4)+2]
    # b = 4..7: A2 * B2[:, 2*((b-4)%4) : 2*((b-4)%4)+2]
    
    # Let's implement this.
    for b in range(8):
        if b < 4:
            # A1: rows 0..2, cols 0..1
            a_row_offset = 0
            a_col_offset = 0
            # B1: rows 0..1, cols 2*(b%4) .. 2*(b%4)+1
            b_row_offset = 0
            b_col_offset = 2 * (b % 4)
        else:
            # A2: rows 0..2, cols 2..3
            a_row_offset = 0
            a_col_offset = 2
            # B2: rows 2..3, cols 2*((b-4)%4) .. 2*((b-4)%4)+1
            b_row_offset = 2
            b_col_offset = 2 * ((b - 4) % 4)
        
        base_col = b * 11
        
        # Strassen for top 2x2 part of A_block (rows 0-1, cols a_col_offset..a_col_offset+1)
        # A_top is 2x2.
        # M1 = (A11+A22)(B11+B22)
        # In U: A11 and A22 are at (0, a_col_offset) and (1, a_col_offset+1)
        # In V: B11 and B22 are at (b_row_offset, b_col_offset) and (b_row_offset+1, b_col_offset+1)
        # In W: C11 is at (b_col_offset, 0), C12 is at (b_col_offset+1, 0), etc.
        
        # Helper to add a term
        def add_term(u_idx, v_idx, w_idx, sign):
            U[u_idx][base_col + term_idx[0]] += sign
            V[v_idx][base_col + term_idx[0]] += sign
            W[w_idx][base_col + term_idx[0]] += sign
        
        # We need to map the 11 terms to specific U, V, W indices.
        # Terms 0-6: Strassen
        # Terms 7-10: Schoolbook for bottom
        
        # Strassen terms:
        # M1: (A11+A22)(B11+B22)
        # U: A11, A22 -> (0, a_col_offset), (1, a_col_offset+1)
        # V: B11, B22 -> (b_row_offset, b_col_offset), (b_row_offset+1, b_col_offset+1)
        # W: C11, C22 -> (b_col_offset, 0), (b_col_offset+1, 1) ? No, W index is (k*n+i).
        # C is 3x8. W[k*n+i] where k is col of C (0..7), i is row of C (0..2).
        # For top 2x2, C rows 0,1. C cols b_col_offset, b_col_offset+1.
        # C11 -> k=b_col_offset, i=0 -> W[b_col_offset*3+0]
        # C12 -> k=b_col_offset+1, i=0 -> W[(b_col_offset+1)*3+0]
        # C21 -> k=b_col_offset, i=1 -> W[b_col_offset*3+1]
        # C22 -> k=b_col_offset+1, i=1 -> W[(b_col_offset+1)*3+1]
        
        # Let's just use a dictionary to map term indices to (U_indices, V_indices, W_indices, signs)
        # This is complex. Let's just iterate and add.
        
        # M1: (A11+A22)(B11+B22)
        # U: 0, a_col_offset; 1, a_col_offset+1
        # V: b_row_offset, b_col_offset; b_row_offset+1, b_col_offset+1
        # W: b_col_offset*3+0 (C11), (b_col_offset+1)*3+1 (C22) ? No.
        # C11 = M1+M4-M5+M7
        # C12 = M3+M5
        # C21 = M2+M4
        # C22 = M1-M2+M3+M6
        
        # Let's define the 7 Strassen terms explicitly.
        # Term 0: M1
        # Term 1: M2
        # Term 2: M3
        # Term 3: M4
        # Term 4: M5
        # Term 5: M6
        # Term 6: M7
        
        # M1 = (A11+A22)(B11+B22)
        U[0 * m + a_col_offset][base_col + 0] += 1
        U[1 * m + a_col_offset + 1][base_col + 0] += 1
        V[b_row_offset * p + b_col_offset][base_col + 0] += 1
        V[(b_row_offset + 1) * p + b_col_offset + 1][base_col + 0] += 1
        W[b_col_offset * n + 0][base_col + 0] += 1  # C11
        W[(b_col_offset + 1) * n + 1][base_col + 0] += 1  # C22
        
        # M2 = (A21+A22)B11
        U[1 * m + a_col_offset][base_col + 1] += 1
        U[1 * m + a_col_offset + 1][base_col + 1] += 1
        V[b_row_offset * p + b_col_offset][base_col + 1] += 1
        W[b_col_offset * n + 1][base_col + 1] += 1  # C21
        W[(b_col_offset + 1) * n + 1][base_col + 1] -= 1  # C22
        
        # M3 = A11(B21-B22)
        U[0 * m + a_col_offset][base_col + 2] += 1
        V[(b_row_offset + 1) * p + b_col_offset][base_col + 2] += 1
        V[(b_row_offset + 1) * p + b_col_offset + 1][base_col + 2] -= 1
        W[(b_col_offset + 1) * n + 0][base_col + 2] += 1  # C12
        W[(b_col_offset + 1) * n + 1][base_col + 2] += 1  # C22
        
        # M4 = A22(B12-B11)
        U[1 * m + a_col_offset + 1][base_col + 3] += 1
        V[b_row_offset * p + b_col_offset + 1][base_col + 3] += 1
        V[b_row_offset * p + b_col_offset][base_col + 3] -= 1
        W[b_col_offset * n + 0][base_col + 3] += 1  # C11
        W[b_col_offset * n + 1][base_col + 3] += 1  # C21
        
        # M5 = (A11+A12)B22
        U[0 * m + a_col_offset][base_col + 4] += 1
        U[0 * m + a_col_offset + 1][base_col + 4] += 1
        V[(b_row_offset + 1) * p + b_col_offset + 1][base_col + 4] += 1
        W[b_col_offset * n + 0][base_col + 4] -= 1  # C11
        W[(b_col_offset + 1) * n + 0][base_col + 4] += 1  # C12
        
        # M6 = (A21-A11)(B11+B12)
        U[1 * m + a_col_offset][base_col + 5] += 1
        U[0 * m + a_col_offset][base_col + 5] -= 1
        V[b_row_offset * p + b_col_offset][base_col + 5] += 1
        V[b_row_offset * p + b_col_offset + 1][base_col + 5] += 1
        W[(b_col_offset + 1) * n + 1][base_col + 5] += 1  # C22
        
        # M7 = (A12-A22)(B21+B22)
        U[0 * m + a_col_offset + 1][base_col + 6] += 1
        U[1 * m + a_col_offset + 1][base_col + 6] -= 1
        V[(b_row_offset + 1) * p + b_col_offset][base_col + 6] += 1
        V[(b_row_offset + 1) * p + b_col_offset + 1][base_col + 6] += 1
        W[b_col_offset * n + 0][base_col + 6] += 1  # C11
        
        # Schoolbook for bottom 1x2 part of A_block (row 2, cols a_col_offset..a_col_offset+1)
        # A_bot is 1x2.
        # C_bot[0] = A_bot[0,0]*B[0,0] + A_bot[0,1]*B[1,0]
        # C_bot[1] = A_bot[0,0]*B[0,1] + A_bot[0,1]*B[1,1]
        
        # Term 7: A_bot[0,0]*B[0,0]*C_bot[0]
        U[2 * m + a_col_offset][base_col + 7] += 1
        V[b_row_offset * p + b_col_offset][base_col + 7] += 1
        W[b_col_offset * n + 2][base_col + 7] += 1
        
        # Term 8: A_bot[0,1]*B[1,0]*C_bot[0]
        U[2 * m + a_col_offset + 1][base_col + 8] += 1
        V[(b_row_offset + 1) * p + b_col_offset][base_col + 8] += 1
        W[b_col_offset * n + 2][base_col + 8] += 1
        
        # Term 9: A_bot[0,0]*B[0,1]*C_bot[1]
        U[2 * m + a_col_offset][base_col + 9] += 1
        V[b_row_offset * p + b_col_offset + 1][base_col + 9] += 1
        W[(b_col_offset + 1) * n + 2][base_col + 9] += 1
        
        # Term 10: A_bot[0,1]*B[1,1]*C_bot[1]
        U[2 * m + a_col_offset + 1][base_col + 10] += 1
        V[(b_row_offset + 1) * p + b_col_offset + 1][base_col + 10] += 1
        W[(b_col_offset + 1) * n + 2][base_col + 10] += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
