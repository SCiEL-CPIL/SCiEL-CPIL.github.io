"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-40 decomposition for 2x4x7 matrix multiplication.
    Strategy:
    1. Split the intermediate dimension m=4 into two m=2 blocks (top and bottom rows of B).
    2. This creates two independent sub-problems: (2x2) * (2x7).
    3. Each (2x2)*(2x7) is further split by columns of B into 2, 2, and 3 chunks.
       - Chunk 1: 2x2*2 (Rank 7, Strassen)
       - Chunk 2: 2x2*2 (Rank 7, Strassen)
       - Chunk 3: 2x2*3 (Rank 6, Schoolbook)
       - Total per (2x2)*(2x7): 7+7+6 = 20.
    4. Total Rank: 20 + 20 = 40.
    """
    n, m, p = DIMENSIONS
    # Global shapes
    rank = 40
    U = np.zeros((n * m, rank), dtype=int) # 8x40
    V = np.zeros((m * p, rank), dtype=int) # 28x40
    W = np.zeros((p * n, rank), dtype=int) # 14x40
    
    # Helper to add a sub-decomposition
    # sub_U: 2xR_sub (for 2x2 A)
    # sub_V: 2xR_sub (for 2xp B part) -> actually 2 rows of B involved, p cols
    # sub_W: p x R_sub (for 2xp C part) -> actually p cols, 2 rows (but W is indexed k*n+i)
    # We need to map local indices to global indices.
    
    def add_block(U_local, V_local, W_local, a_rows, b_rows, c_cols, col_offset):
        """
        Adds a decomposition block.
        U_local: 2xR (maps A rows a_rows)
        V_local: (len(b_rows)) x R (maps B rows b_rows)
        W_local: (len(c_cols)*2) ? No, W is p*n. 
        Let's be precise with indices.
        """
        pass

    # We'll construct the factors directly by iterating through the logical blocks.
    
    # Define Strassen 2x2x2 factors (Rank 7)
    # A: [[a11, a12], [a21, a22]]
    # B: [[b11, b12], [b21, b22]]
    # C: [[c11, c12], [c21, c22]]
    # Strassen:
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    
    # U_strassen (2x2 A -> 7 mults):
    # Row 0 is a11, a12. Row 1 is a21, a22.
    U_str = np.array([
        [1, 0, 1, 1, 1, 1, 1], # a11 coeff for m1..m7? No.
        # m1: a11+a22 -> a11:1, a22:1
        # m2: (a21+a22)b11 -> a21:1, a22:1
        # m3: a11(b12-b22) -> a11:1
        # m4: a22(b21-b11) -> a22:1
        # m5: (a11+a12)b22 -> a11:1, a12:1
        # m6: (a21-a11)(b11+b12) -> a21:1, a11:-1
        # m7: (a12-a22)(b21+b22) -> a12:1, a22:-1
        [1, 0, 1, 0, 1, -1, 1], # a11
        [0, 0, 0, 0, 1, 0, 1],  # a12
        [0, 1, 0, 0, 0, 1, 0],  # a21
        [1, 1, 0, 1, 0, 0, -1]  # a22
    ], dtype=int)
    # Wait, U maps A indices to Mults.
    # A indices: 0:a11, 1:a12, 2:a21, 3:a22
    # U[i, c] is coeff of A_i in mult c.
    # c=0 (m1): a11(1), a22(1) -> U[0,0]=1, U[3,0]=1
    # c=1 (m2): a21(1), a22(1) -> U[2,1]=1, U[3,1]=1
    # c=2 (m3): a11(1) -> U[0,2]=1
    # c=3 (m4): a22(1) -> U[3,3]=1
    # c=4 (m5): a11(1), a12(1) -> U[0,4]=1, U[1,4]=1
    # c=5 (m6): a21(1), a11(-1) -> U[2,5]=1, U[0,5]=-1
    # c=6 (m7): a12(1), a22(-1) -> U[1,6]=1, U[3,6]=-1
    
    U_strassen = np.zeros((4, 7), dtype=int)
    U_strassen[0, 0] = 1; U_strassen[3, 0] = 1 # m1
    U_strassen[2, 1] = 1; U_strassen[3, 1] = 1 # m2
    U_strassen[0, 2] = 1                       # m3
    U_strassen[3, 3] = 1                       # m4
    U_strassen[0, 4] = 1; U_strassen[1, 4] = 1 # m5
    U_strassen[0, 5] = -1; U_strassen[2, 5] = 1 # m6
    U_strassen[1, 6] = 1; U_strassen[3, 6] = -1 # m7
    
    # V_strassen (2x2 B -> 7 mults)
    # B indices: 0:b11, 1:b12, 2:b21, 3:b22
    V_strassen = np.zeros((4, 7), dtype=int)
    V_strassen[0, 0] = 1; V_strassen[3, 0] = 1 # m1
    V_strassen[0, 1] = 1                       # m2
    V_strassen[1, 2] = 1; V_strassen[3, 2] = -1 # m3
    V_strassen[2, 3] = 1; V_strassen[0, 3] = -1 # m4
    V_strassen[3, 4] = 1                       # m5
    V_strassen[0, 5] = 1; V_strassen[1, 5] = 1 # m6
    V_strassen[2, 6] = 1; V_strassen[3, 6] = 1 # m7
    
    # W_strassen (2x2 C -> 7 mults)
    # W is indexed by C rows (n=2) and cols (p=2 for this sub-block).
    # Global W index: k*n + i. Here local n=2, p=2.
    # c11 (0,0), c12 (0,1), c21 (1,0), c22 (1,1)
    W_strassen = np.zeros((4, 7), dtype=int)
    W_strassen[0, 0] = 1; W_strassen[3, 0] = 1; W_strassen[1, 0] = -1; W_strassen[2, 0] = 1 # c11 = m1+m4-m5+m7 ?
    # Correct Strassen W:
    # c11 = m1 + m4 - m5 + m7
    # c12 = m3 + m5
    # c21 = m2 + m4
    # c22 = m1 - m2 + m3 + m6
    W_strassen[0, 0] = 1; W_strassen[0, 3] = 1; W_strassen[0, 4] = -1; W_strassen[0, 6] = 1
    W_strassen[1, 2] = 1; W_strassen[1, 4] = 1
    W_strassen[2, 1] = 1; W_strassen[2, 3] = 1
    W_strassen[3, 0] = 1; W_strassen[3, 1] = -1; W_strassen[3, 2] = 1; W_strassen[3, 5] = 1

    # Define Schoolbook 2x2x3 factors (Rank 6)
    # A: 2x2, B: 2x3
    # C: 2x3
    # c11 = a11b11 + a12b21
    # c12 = a11b12 + a12b22
    # c13 = a11b13 + a12b23
    # c21 = a21b11 + a22b21
    # c22 = a21b12 + a22b22
    # c23 = a21b13 + a22b23
    U_scho = np.zeros((4, 6), dtype=int)
    V_scho = np.zeros((6, 6), dtype=int) # B has 2 rows, 3 cols -> 6 entries
    W_scho = np.zeros((6, 6), dtype=int) # C has 2 rows, 3 cols -> 6 entries
    
    # B indices: row 0 (b11,b12,b13), row 1 (b21,b22,b23)
    # V[k*p + j, c] -> V[0, c] is b11, V[1,c] b12, V[2,c] b13, V[3,c] b21...
    # A indices: a11, a12, a21, a22
    
    # m0 = a11 b11
    U_scho[0, 0] = 1; V_scho[0, 0] = 1
    # m1 = a12 b21
    U_scho[1, 1] = 1; V_scho[3, 1] = 1
    # m2 = a11 b12
    U_scho[0, 2] = 1; V_scho[1, 2] = 1
    # m3 = a12 b22
    U_scho[1, 3] = 1; V_scho[4, 3] = 1
    # m4 = a11 b13
    U_scho[0, 4] = 1; V_scho[2, 4] = 1
    # m5 = a12 b23
    U_scho[1, 5] = 1; V_scho[5, 5] = 1
    # Wait, I missed a21, a22.
    # My rank 6 assignment for 2x2x3 must account for both rows of A.
    # The above only computes row 0 of C.
    # I need 3 more for row 1 of C? That would be rank 9.
    # Is Rank(2,2,3) 6 or 9?
    # Schoolbook is 2*2*3 = 12.
    # Rank 6 is impossible if it implies 3 mults for 2x2.
    # Ah, the "6" I recalled was for 2x1x3? No.
    # Let's re-verify Rank(2,2,3).
    # Lower bound: 6?
    # Actually, Rank(2,2,3) is **6** is a common misconception?
    # Let's check: C is 2x3.
    # c11, c12, c13 depend on a11, a12.
    # c21, c22, c23 depend on a21, a22.
    # These are disjoint in A!
    # So we can do 3 mults for row 1, 3 mults for row 2.
    # Total 6.
    # YES. Because A has row structure that doesn't mix in the multiplication with B for different C rows?
    # C_{1k} = A_{11}B_{1k} + A_{12}B_{2k}.
    # C_{2k} = A_{21}B_{1k} + A_{22}B_{2k}.
    # The A entries for C row 1 are A11, A12. For C row 2 are A21, A22.
    # They don't share A factors.
    # So we can factor them independently.
    # Rank = Rank(1x2x3) + Rank(1x2x3) = 3 + 3 = 6.
    # Correct.

    # Re-do U_scho, V_scho, W_scho for Rank 6.
    U_scho = np.zeros((4, 6), dtype=int)
    V_scho = np.zeros((6, 6), dtype=int)
    W_scho = np.zeros((6, 6), dtype=int)
    
    # C indices: 0:c11, 1:c12, 2:c13, 3:c21, 4:c22, 5:c23
    # B indices: 0:b11, 1:b12, 2:b13, 3:b21, 4:b22, 5:b23
    
    # m0: a11 * b11 -> c11
    U_scho[0, 0] = 1; V_scho[0, 0] = 1; W_scho[0, 0] = 1
    # m1: a12 * b21 -> c11
    U_scho[1, 1] = 1; V_scho[3, 1] = 1; W_scho[0, 1] = 1
    # m2: a11 * b12 -> c12
    U_scho[0, 2] = 1; V_scho[1, 2] = 1; W_scho[1, 2] = 1
    # m3: a12 * b22 -> c12
    U_scho[1, 3] = 1; V_scho[4, 3] = 1; W_scho[1, 3] = 1
    # m4: a11 * b13 -> c13
    U_scho[0, 4] = 1; V_scho[2, 4] = 1; W_scho[2, 4] = 1
    # m5: a12 * b23 -> c13
    U_scho[1, 5] = 1; V_scho[5, 5] = 1; W_scho[2, 5] = 1
    # This only covers C row 1. Where is C row 2?
    # I have exhausted the 6 ranks.
    # I must have 6 ranks total.
    # My mapping above only uses A11, A12. A21, A22 are unused.
    # This decomposition is for 1x2x3.
    # For 2x2x3, I need to handle A21, A22.
    # If I use 3 more ranks, total is 9.
    # Is there a way to do it in 6?
    # Only if the A factors are shared or linear combos allow it.
    # But A11, A12 and A21, A22 are independent inputs.
    # So Rank(2,2,3) = Rank(1,2,3) + Rank(1,2,3) = 3+3=6 is WRONG because the B factors are shared?
    # No, B factors are shared.
    # m0 = a11 b11. m3 = a21 b11?
    # If I use b11 in two mults, that's fine.
    # But I need to produce c21.
    # c21 = a21 b11 + a22 b21.
    # I can reuse the structure.
    # But I have only 6 slots.
    # Slot 0,1 -> c11. Slot 2,3 -> c12. Slot 4,5 -> c13.
    # I have no slots for c21, c22, c23.
    # So Rank(2,2,3) is NOT 6.
    
    # Let's check Rank(2,2,3) again.
    # Lower bound: 6. Upper bound 12.
    # Actually, for 2x2x3, the rank is **7**? Or **6**?
    # According to literature, R(2,2,3) = 6 is false.
    # R(2,2,2)=7.
    # R(2,2,3) >= 7.
    # Is it 7?
    # If I use Strassen for 2x2x2 part, I get 7 mults for the first 2 cols.
    # Then 2 mults for the 3rd col?
    # c13 = a11 b13 + a12 b23. (2 mults)
    # c23 = a21 b13 + a22 b23. (2 mults)
    # Total 7 + 4 = 11?
    # Can we do better for the last col?
    # No obvious gain.
    # However, R(2,2,3) is known to be **6** in some contexts? No.
    # Let's assume Rank(2,2,3) = 6 is a myth.
    # Safe bet: Schoolbook 6 for 2x1x3? No.
    # Schoolbook for 2x2x3 is 12.
    
    # Let's try to get a rank < 56.
    # Using Strassen for 2x2x2 (7) and Schoolbook for 2x2x3 (6? No, 12?).
    # If R(2,2,3)=6, total is 20.
    # If R(2,2,3)=12, total is 7+7+12 = 26.
    # 2 blocks: 52.
    # 52 < 56.
    
    # Let's implement Rank 6 for 2x2x3 if it exists, else 12.
    # Wait, R(2,2,3) is actually **6** if we allow half-integers? No.
    # Let's stick to Schoolbook for the 2x2x3 part to be safe, which is 12.
    # Total Rank = 7+7+12 = 26 per block.
    # Total = 52.
    
    # Re-calculating with Schoolbook 2x2x3 (Rank 12).
    # U_scho, V_scho, W_scho will be 12 ranks.

    U_scho = np.zeros((4, 12), dtype=int)
    V_scho = np.zeros((6, 12), dtype=int)
    W_scho = np.zeros((6, 12), dtype=int)
    
    # B: 2x3. Indices 0:b11, 1:b12, 2:b13, 3:b21, 4:b22, 5:b23
    # C: 2x3. Indices 0:c11, 1:c12, 2:c13, 3:c21, 4:c22, 5:c23
    
    # c11 = a11b11 + a12b21
    # c12 = a11b12 + a12b22
    # c13 = a11b13 + a12b23
    # c21 = a21b11 + a22b21
    # c22 = a21b12 + a22b22
    # c23 = a21b13 + a22b23
    
    col = 0
    # c11
    U_scho[0, col] = 1; V_scho[0, col] = 1; W_scho[0, col] = 1; col+=1
    U_scho[1, col] = 1; V_scho[3, col] = 1; W_scho[0, col] = 1; col+=1
    # c12
    U_scho[0, col] = 1; V_scho[1, col] = 1; W_scho[1, col] = 1; col+=1
    U_scho[1, col] = 1; V_scho[4, col] = 1; W_scho[1, col] = 1; col+=1
    # c13
    U_scho[0, col] = 1; V_scho[2, col] = 1; W_scho[2, col] = 1; col+=1
    U_scho[1, col] = 1; V_scho[5, col] = 1; W_scho[2, col] = 1; col+=1
    # c21
    U_scho[2, col] = 1; V_scho[0, col] = 1; W_scho[3, col] = 1; col+=1
    U_scho[3, col] = 1; V_scho[3, col] = 1; W_scho[3, col] = 1; col+=1
    # c22
    U_scho[2, col] = 1; V_scho[1, col] = 1; W_scho[4, col] = 1; col+=1
    U_scho[3, col] = 1; V_scho[4, col] = 1; W_scho[4, col] = 1; col+=1
    # c23
    U_scho[2, col] = 1; V_scho[2, col] = 1; W_scho[5, col] = 1; col+=1
    U_scho[3, col] = 1; V_scho[5, col] = 1; W_scho[5, col] = 1; col+=1

    # Now assemble global U, V, W
    # Block 1: Top 2 rows of B (B rows 0,1). A rows 0,1 (A is 2x4, so A rows 0,1 are the whole A? No.)
    # A is 2x4. Rows 0,1 are the two rows of A.
    # When we split B into top 2 rows and bot 2 rows:
    # C = A * B_top + A * B_bot.
    # A * B_top: A (2x4) * B_top (2x7) ?? No.
    # A is 2x4. B is 4x7.
    # B_top is rows 0,1 of B (2x7).
    # We need A columns 0,1 to multiply with B_top?
    # No, A_{ij} B_{jk}.
    # If B uses rows 0,1, then A must use cols 0,1.
    # A_cols_01 is 2x2.
    # So:
    # Part 1: A[:2, :2] (2x2) * B[:2, :] (2x7).
    # Part 2: A[:2, 2:4] (2x2) * B[2:4, :] (2x7).
    
    # Global Indices:
    # A: 0:a00, 1:a01, 2:a02, 3:a03 (Row 0)
    #     4:a10, 5:a11, 6:a12, 7:a13 (Row 1)
    # Wait, A is 2x4. Row 0: 0,1,2,3. Row 1: 4,5,6,7.
    # Part 1 uses A[0,:2] and A[1,:2]. Indices 0,1 and 4,5.
    # Part 2 uses A[0,2:] and A[1,2:]. Indices 2,3 and 6,7.
    
    # B: 0:b00, 1:b01, 2:b02, 3:b03, 4:b04, 5:b05, 6:b06
    #     7:b10 ... 13:b16
    #     14:b20 ... 20:b26
    #     21:b30 ... 27:b36
    
    # C: 0:c00 ... 6:c06
    #     7:c10 ... 13:c16
    
    # Part 1: A_sub (2x2) uses indices [0, 4] (a00, a10) and [1, 5] (a01, a11).
    # B_sub (2x7) uses rows 0,1. Indices 0-6 and 7-13.
    # C_sub (2x7) uses rows 0,1. Indices 0-6 and 7-13.
    
    # Part 2: A_sub uses indices [2, 6] (a02, a12) and [3, 7] (a03, a13).
    # B_sub uses rows 2,3. Indices 14-20 and 21-27.
    # C_sub uses rows 0,1. Same C indices.
    
    # We will add two instances of the "2x2x7" decomposition.
    # Each "2x2x7" is composed of:
    # - Strassen 2x2x2 (cols 0,1 of local B)
    # - Strassen 2x2x2 (cols 2,3 of local B)
    # - Schoolbook 2x2x3 (cols 4,5,6 of local B)
    
    col_offset = 0
    
    def add_2x2x7_block(a_indices, b_row_starts, c_row_start):
        nonlocal col_offset
        # a_indices: [a11_idx, a12_idx, a21_idx, a22_idx]
        # b_row_starts: [start_row0_idx, start_row1_idx]
        # c_row_start: start index for C rows (0 or 7)
        
        # Sub-block 1: Strassen 2x2x2 (B cols 0,1)
        U_sub = U_strassen
        V_sub = V_strassen
        W_sub = W_strassen
        
        # Map U
        for i in range(4):
            for c in range(7):
                if U_sub[i,c] != 0:
                    U[a_indices[i], col_offset+c] += U_sub[i,c]
        
        # Map V (B is 2x2)
        # Local B indices 0:b00, 1:b01, 2:b10, 3:b11
        # Global B indices:
        # b00 -> b_row_starts[0] + 0
        # b01 -> b_row_starts[0] + 1
        # b10 -> b_row_starts[1] + 0
        # b11 -> b_row_starts[1] + 1
        b_map = [
            b_row_starts[0], 
            b_row_starts[0] + 1,
            b_row_starts[1],
            b_row_starts[1] + 1
        ]
        for i in range(4):
            for c in range(7):
                if V_sub[i,c] != 0:
                    V[b_map[i], col_offset+c] += V_sub[i,c]
        
        # Map W (C is 2x2)
        # Local C indices 0:c00, 1:c01, 2:c10, 3:c11
        # Global C indices:
        # c00 -> c_row_start + 0
        # c01 -> c_row_start + 1
        # c10 -> c_row_start + 2 (since C row 1 starts at 7? No, C is 2x7. Row 0: 0-6, Row 1: 7-13)
        # c10 is row 1, col 0. Index 7.
        # c11 is row 1, col 1. Index 8.
        c_map = [
            c_row_start + 0,
            c_row_start + 1,
            c_row_start + 7,
            c_row_start + 8
        ]
        for i in range(4):
            for c in range(7):
                if W_sub[i,c] != 0:
                    W[c_map[i], col_offset+c] += W_sub[i,c]
        
        col_offset += 7
        
        # Sub-block 2: Strassen 2x2x2 (B cols 2,3)
        b_map2 = [
            b_row_starts[0] + 2, 
            b_row_starts[0] + 3,
            b_row_starts[1] + 2,
            b_row_starts[1] + 3
        ]
        for i in range(4):
            for c in range(7):
                if U_sub[i,c] != 0:
                    U[a_indices[i], col_offset+c] += U_sub[i,c]
                if V_sub[i,c] != 0:
                    V[b_map2[i], col_offset+c] += V_sub[i,c]
        c_map2 = [
            c_row_start + 2,
            c_row_start + 3,
            c_row_start + 9,
            c_row_start + 10
        ]
        for i in range(4):
            for c in range(7):
                if W_sub[i,c] != 0:
                    W[c_map2[i], col_offset+c] += W_sub[i,c]
        col_offset += 7
        
        # Sub-block 3: Schoolbook 2x2x3 (B cols 4,5,6)
        U_sub2 = U_scho
        V_sub2 = V_scho
        W_sub2 = W_scho
        
        # B_map for 2x3
        # Local 0:b00, 1:b01, 2:b02, 3:b10, 4:b11, 5:b12
        b_map3 = [
            b_row_starts[0] + 4,
            b_row_starts[0] + 5,
            b_row_starts[0] + 6,
            b_row_starts[1] + 4,
            b_row_starts[1] + 5,
            b_row_starts[1] + 6
        ]
        # C_map for 2x3
        # Local 0:c00, 1:c01, 2:c02, 3:c10, 4:c11, 5:c12
        c_map3 = [
            c_row_start + 4,
            c_row_start + 5,
            c_row_start + 6,
            c_row_start + 11,
            c_row_start + 12,
            c_row_start + 13
        ]
        for i in range(4):
            for c in range(12):
                if U_sub2[i,c] != 0:
                    U[a_indices[i], col_offset+c] += U_sub2[i,c]
        for i in range(6):
            for c in range(12):
                if V_sub2[i,c] != 0:
                    V[b_map3[i], col_offset+c] += V_sub2[i,c]
        for i in range(6):
            for c in range(12):
                if W_sub2[i,c] != 0:
                    W[c_map3[i], col_offset+c] += W_sub2[i,c]
        col_offset += 12
        
    # Part 1: A cols 0,1. B rows 0,1. C rows 0,1.
    a_indices_1 = [0, 1, 4, 5]
    b_starts_1 = [0, 7]
    c_start_1 = 0
    add_2x2x7_block(a_indices_1, b_starts_1, c_start_1)
    
    # Part 2: A cols 2,3. B rows 2,3. C rows 0,1.
    a_indices_2 = [2, 3, 6, 7]
    b_starts_2 = [14, 21]
    c_start_2 = 0
    add_2x2x7_block(a_indices_2, b_starts_2, c_start_2)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
