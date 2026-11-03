"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 * 14 = 56
    rank = 56
    U = [[0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 16 x 56
    
    # We process 4 blocks of B, each 4x2.
    # B is 4x8. Columns 0-1, 2-3, 4-5, 6-7.
    # For each block, we do Strassen 2x2x2 twice (left and right of A).
    
    # Strassen coefficients for 2x2 multiplication
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # A is 2x4. A_left is 2x2 (cols 0,1), A_right is 2x2 (cols 2,3).
    # B_block is 4x2. B_top is 2x2 (rows 0,1), B_bot is 2x2 (rows 2,3).
    # C = A_left B_top + A_right B_bot
    
    # Indices for U (A entries)
    # A[0,0]=0, A[0,1]=1, A[0,2]=2, A[0,3]=3
    # A[1,0]=4, A[1,1]=5, A[1,2]=6, A[1,3]=7
    
    # Indices for V (B entries)
    # B[0,0]=0, B[0,1]=1
    # B[1,0]=2, B[1,1]=3
    # B[2,0]=4, B[2,1]=5
    # B[3,0]=6, B[3,1]=7
    # Block 0: cols 0,1. Block 1: cols 2,3. Block 2: cols 4,5. Block 3: cols 6,7.
    
    # Indices for W (C entries)
    # C[0,0]=0, C[0,1]=1
    # C[1,0]=2, C[1,1]=3
    # Block 0: cols 0,1. Block 1: cols 2,3. Block 2: cols 4,5. Block 3: cols 6,7.
    
    def add_strassen_term(col_idx, a_left, a_right, b_top, b_bot, c_block_idx):
        # a_left, a_right are lists of 2 indices each? No, they are 2x2 matrices of indices.
        # Let's just hardcode the 14 terms for one block.
        pass

    # We will generate the 14 terms for each of the 4 blocks.
    # Block b (0..3)
    # A is fixed.
    # B_block_b has cols 2b, 2b+1.
    # V indices:
    # B[0, 2b], B[0, 2b+1] -> 0, 1
    # B[1, 2b], B[1, 2b+1] -> 2, 3
    # B[2, 2b], B[2, 2b+1] -> 4, 5
    # B[3, 2b], B[3, 2b+1] -> 6, 7
    # So V indices are 2b, 2b+1, 2+2b, 3+2b, 4+2b, 5+2b, 6+2b, 7+2b.
    
    # C indices:
    # C[0, 2b], C[0, 2b+1] -> 2b, 2b+1
    # C[1, 2b], C[1, 2b+1] -> 2+2b, 3+2b
    
    # A indices:
    # A_left: (0,1), (4,5)
    # A_right: (2,3), (6,7)
    
    # Strassen for A_left B_top
    # A00=0, A01=1, A10=4, A11=5
    # B00=2b, B01=2b+1, B10=2+2b, B11=3+2b
    
    # Strassen for A_right B_bot
    # A00=2, A01=3, A10=6, A11=7
    # B00=4+2b, B01=5+2b, B10=6+2b, B11=7+2b
    
    col = 0
    for b in range(4):
        # 14 terms
        # 1. (A00+A11)(B00+B11)
        U[0][col] = 1; U[5][col] = 1
        V[2*b][col] = 1; V[3+2*b][col] = 1
        W[2*b][col] = 1; W[3+2*b][col] = 1
        col += 1
        
        # 2. (A10+A11)B00
        U[4][col] = 1; U[5][col] = 1
        V[2*b][col] = 1
        W[2+2*b][col] = 1; W[3+2*b][col] = 1
        col += 1
        
        # 3. A00(B01-B11)
        U[0][col] = 1
        V[2*b+1][col] = 1; V[3+2*b][col] = -1
        W[2*b+1][col] = 1
        col += 1
        
        # 4. A11(B10-B00)
        U[5][col] = 1
        V[2+2*b][col] = 1; V[2*b][col] = -1
        W[2*b][col] = 1; W[3+2*b][col] = 1
        col += 1
        
        # 5. (A00+A01)B11
        U[0][col] = 1; U[1][col] = 1
        V[3+2*b][col] = 1
        W[2*b][col] = -1; W[2*b+1][col] = 1
        col += 1
        
        # 6. (A10-A00)(B00+B01)
        U[4][col] = 1; U[0][col] = -1
        V[2*b][col] = 1; V[2*b+1][col] = 1
        W[3+2*b][col] = 1
        col += 1
        
        # 7. (A01-A11)(B10+B11)
        U[1][col] = 1; U[5][col] = -1
        V[2+2*b][col] = 1; V[3+2*b][col] = 1
        W[2*b][col] = 1; W[3+2*b][col] = 1
        col += 1
        
        # Now A_right B_bot
        # A00=2, A01=3, A10=6, A11=7
        # B00=4+2b, B01=5+2b, B10=6+2b, B11=7+2b
        
        # 8. (A00+A11)(B00+B11)
        U[2][col] = 1; U[7][col] = 1
        V[4+2*b][col] = 1; V[7+2*b][col] = 1
        W[2*b][col] = 1; W[3+2*b][col] = 1
        col += 1
        
        # 9. (A10+A11)B00
        U[6][col] = 1; U[7][col] = 1
        V[4+2*b][col] = 1
        W[2+2*b][col] = 1; W[3+2*b][col] = 1
        col += 1
        
        # 10. A00(B01-B11)
        U[2][col] = 1
        V[5+2*b][col] = 1; V[7+2*b][col] = -1
        W[2*b+1][col] = 1
        col += 1
        
        # 11. A11(B10-B00)
        U[7][col] = 1
        V[6+2*b][col] = 1; V[4+2*b][col] = -1
        W[2*b][col] = 1; W[3+2*b][col] = 1
        col += 1
        
        # 12. (A00+A01)B11
        U[2][col] = 1; U[3][col] = 1
        V[7+2*b][col] = 1
        W[2*b][col] = -1; W[2*b+1][col] = 1
        col += 1
        
        # 13. (A10-A00)(B00+B01)
        U[6][col] = 1; U[2][col] = -1
        V[4+2*b][col] = 1; V[5+2*b][col] = 1
        W[3+2*b][col] = 1
        col += 1
        
        # 14. (A01-A11)(B10+B11)
        U[3][col] = 1; U[7][col] = -1
        V[6+2*b][col] = 1; V[7+2*b][col] = 1
        W[2*b][col] = 1; W[3+2*b][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
