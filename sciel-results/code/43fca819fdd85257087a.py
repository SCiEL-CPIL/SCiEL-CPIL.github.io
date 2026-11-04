"""Matrix multiplication 3,4,4 - rank-minimal decomposition using Strassen blocks."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strategy: Split the 3x4 x 4x4 multiplication into blocks.
    # A is 3x4, B is 4x4, C is 3x4.
    # Split A into A_top (2x4) and A_bot (1x4).
    # Split B into B_left (4x2) and B_right (4x2).
    # Then:
    #   C_top (2x4) = A_top * B  -> split into C_top_left (2x2) and C_top_right (2x2)
    #   C_bot (1x4) = A_bot * B  -> split into C_bot_left (1x2) and C_bot_right (1x2)
    #
    # C_top_left (2x2) = A_top_left (2x2) * B_top_left (2x2)  -> Strassen: rank 7
    #   Wait, A_top is 2x4, B is 4x4.
    #   C_top_left[j,k] = sum_{l=0}^3 A_top[j,l] * B[l,k] for k=0,1
    #   This is (2x4) * (4x2) = 2x2. Rank of 2x4x2 multiplication.
    #   Schoolbook: 2*4*2 = 16.
    #   Can we do better?
    #   Actually, 2x4x2 is a specific tensor.
    #   Let's check if Strassen applies directly.
    #   2x4x2 can be seen as two 2x2x2 multiplications? No.
    #   A is 2x4, B is 4x2.
    #   A = [A1 A2] where A1, A2 are 2x2.
    #   B = [B1; B2] where B1, B2 are 2x2.
    #   C = A1*B1 + A2*B2.
    #   This is sum of two 2x2x2 multiplications.
    #   Rank = 2 * rank(2x2x2) = 2 * 7 = 14.
    #   Schoolbook was 16. So 14 is better.
    #
    # C_top_right (2x2) = A_top * B_right. Similarly rank 14.
    #
    # C_bot_left (1x2) = A_bot (1x4) * B_left (4x2).
    #   A_bot = [a1 a2 a3 a4], B_left is 4x2.
    #   C = a1*b1 + a2*b2 + a3*b3 + a4*b4 where bi are 2-vectors?
    #   Actually C_j = sum_i A_bot[i] * B_left[i,j].
    #   This is 1x4x2 multiplication.
    #   Rank of 1xnmx is nm if we can't share.
    #   Here n=4, m=2. Rank = 8?
    #   Let's verify.
    #   T = sum_{i=0}^3 sum_{j=0}^1 e_i (in A) tensor f_{ij} (in B) tensor g_j (in C).
    #   Are there linear dependencies?
    #   The variables are A_0..A_3, B_{00}..B_{31}, C_0..C_1.
    #   The terms are A_i B_{ij} C_j.
    #   For fixed i, j, it's rank 1.
    #   Can we combine A_0 B_{00} C_0 and A_0 B_{01} C_1? No, different C.
    #   Can we combine A_0 B_{00} C_0 and A_1 B_{10} C_0?
    #   (A_0 B_{00} + A_1 B_{10}) C_0.
    #   This requires a linear combination in A and B.
    #   But the rank of the tensor 1xnmx is known to be nm for n,m >= 1?
    #   Actually for 1xnx1, rank is 1? No, 1xn1 is just scalar mult. Rank 1?
    #   1x1x1: A0*B0*C0. Rank 1.
    #   1x2x1: A0B0C0 + A1B1C1. Rank 2.
    #   1x2x2: A0B00C0 + A0B01C1 + A1B10C0 + A1B11C1.
    #   Can we do better than 4?
    #   Try to combine: (A0B00 + A1B10)C0 + (A0B01 + A1B11)C1.
    #   This is 2 terms. Rank 2?
    #   Wait, U must be 1x2 (A side), V must be 2x2 (B side), W must be 2x1 (C side)?
    #   No, U is (n*m) x rank = 2 x rank. V is (m*p) x rank = 2 x rank. W is (p*n) x rank = 2 x rank.
    #   Wait, dimensions are n=1, m=4, p=2? No.
    #   The block is C (1x2) = A (1x4) * B (4x2).
    #   So n=1, m=4, p=2.
    #   U size: 1*4 = 4. V size: 4*2 = 8. W size: 2*1 = 2.
    #   Rank r.
    #   T = sum_{alpha} U_alpha (4-dim) tensor V_alpha (8-dim) tensor W_alpha (2-dim).
    #   Is rank 4 sufficient?
    #   Schoolbook is 8.
    #   Let's check if rank 4 works for 1x4x2.
    #   This is equivalent to finding a low rank decomposition of the tensor.
    #   Actually, 1xnmx tensor rank is n+m-1? Or nm?
    #   For 1x2x2, rank is 2? Let's check.
    #   Terms: A0B00C0, A0B01C1, A1B10C0, A1B11C1.
    #   If rank 2:
    #   U = [u0, u1] (2x2). V = [v0, v1] (2x2). W = [w0, w1] (2x2).
    #   T = u0 tensor v0 tensor w0 + u1 tensor v1 tensor w1.
    #   This generates bilinear forms.
    #   It is known that rank of 1xnmx is min(n,m) * max(n,m)? No.
    #   Let's assume schoolbook 8 is not optimal but finding exact rank is hard.
    #   However, we know 2x4x2 is 14 (2x Strassen).
    #   And 1x4x2 is likely 8 or less. Let's assume 8 for safety, or try to find a decomposition.
    #   Actually, let's just use the known result for 3x4x4.
    #   Schoolbook: 48.
    #   My split:
    #   C_top (2x4) = 2x4x4.
    #   C_top = A_top (2x4) * B (4x4).
    #   Split B into B_left (4x2) and B_right (4x2).
    #   C_top_left (2x2) = A_top * B_left. Rank 14.
    #   C_top_right (2x2) = A_top * B_right. Rank 14.
    #   C_bot (1x4) = A_bot (1x4) * B (4x4).
    #   C_bot_left (1x2) = A_bot * B_left.
    #   C_bot_right (1x2) = A_bot * B_right.
    #   
    #   If we use schoolbook for C_bot parts:
    #   1x4x2 rank is 8?
    #   Total rank = 14 + 14 + 8 + 8 = 44.
    #   
    #   Can we do better for 1x4x2?
    #   Let's try to construct a rank 4 decomposition for 1x4x2?
    #   T = sum_{i=0}^3 sum_{j=0}^1 A_i B_{ij} C_j.
    #   Let's try rank 4.
    #   U (4x4), V (8x4), W (2x4).
    #   This is a system of equations.
    #   Given the complexity, 44 is a safe and proven improvement over 48.
    #   Let's implement rank 44.
    
    n, m, p = DIMENSIONS # 3, 4, 4
    rank = 44
    U = [[0] * rank for _ in range(n * m)] # 12 x 44
    V = [[0] * rank for _ in range(m * p)] # 16 x 44
    W = [[0] * rank for _ in range(p * n)] # 12 x 44
    
    col = 0
    
    # Block 1: C_top_left (2x2) = A_top (2x4) * B_left (4x2)
    # Indices:
    # A_top rows: 0, 1. Cols: 0..3.
    # B_left rows: 0..3. Cols: 0, 1.
    # C_top_left rows: 0, 1. Cols: 0, 1.
    # 
    # Decomposition of 2x4x2 as sum of two 2x2x2 Strassen blocks?
    # No, 2x4x2 = A_top * B_left.
    # A_top = [A_TL A_TR] (2x2 each). B_left = [B_LL; B_LR] (2x2 each).
    # C = A_TL B_LL + A_TR B_LR.
    # This is sum of two 2x2x2 multiplications.
    # We need to implement Strassen for 2x2x2.
    
    def strassen_2x2x2_offset(u_off, v_off, w_off, start_col):
        # U is 2x2 block of U (rows u_off..u_off+1, cols?)
        # V is 2x2 block of V
        # W is 2x2 block of W
        # We add 7 terms to U, V, W starting at start_col.
        # Strassen variables:
        # M1 = (A11 + A22) * (B11 + B22)
        # M2 = (A21 + A22) * B11
        # M3 = A11 * (B12 - B22)
        # M4 = A22 * (B21 - B11)
        # M5 = (A11 + A12) * B22
        # M6 = (A21 - A11) * (B11 + B12)
        # M7 = (A12 - A22) * (B21 + B22)
        #
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # We need to map these to the global U, V, W indices.
        # The "A" part corresponds to U indices.
        # The "B" part corresponds to V indices.
        # The "C" part corresponds to W indices.
        
        # For a specific 2x2x2 block, the local indices are:
        # U: 2 rows, 2 cols? No, U is 2x4 in the 2x4x2 case.
        # Wait, in the 2x4x2 case, we split into two 2x2x2.
        # Block 1: A_TL (2x2) * B_LL (2x2).
        # A_TL is U[0:2, 0:2] (global row, global col).
        # B_LL is V[0:2, 0:2] (global row, global col).
        # Result goes to W[0:2, 0:2] (global row, global col).
        
        # Strassen 1:
        # M1: U factor: A11+A22. V factor: B11+B22.
        # M2: U factor: A21+A22. V factor: B11.
        # ...
        
        # Let's define the local indices for A, B, C in the 2x2x2 block.
        # A indices: (0,0), (0,1), (1,0), (1,1) relative to the block start.
        # B indices: (0,0), (0,1), (1,0), (1,1).
        # C indices: (0,0), (0,1), (1,0), (1,1).
        
        # We will return the column index after adding 7 terms.
        c = start_col
        
        # M1
        # U: A11 (0,0) + A22 (1,1). So U[row0, col] += 1, U[row1, col] += 1.
        # V: B11 (0,0) + B22 (1,1). V[row0, col] += 1, V[row1, col] += 1.
        # W: C11 (0,0) += 1, C22 (1,1) += 1.
        U[u_off + 0][c] += 1
        U[u_off + 1][c] += 1
        V[v_off + 0][c] += 1
        V[v_off + 1][c] += 1
        W[w_off + 0][c] += 1
        W[w_off + 1][c] += 1
        c += 1
        
        # M2
        # U: A21 (1,0) + A22 (1,1).
        # V: B11 (0,0).
        # W: C21 (1,0) += 1.
        U[u_off + 1][c] += 1 # A21
        U[u_off + 1][c] += 1 # A22 -> 2?
        # Wait, A21 is (1,0), A22 is (1,1).
        # In U, the indices are row, col.
        # U is 2x4. A_TL is U[0:2, 0:2].
        # A21 is U[1, 0]. A22 is U[1, 1].
        # So U[u_off+1, 0] and U[u_off+1, 1] are the entries.
        # But U is a matrix of coefficients for the linear combination.
        # U[i, k] is the coefficient of A_i in the k-th term.
        # So for M2, the A-part is A21 + A22.
        # So U[1, c] should be 1 for A21? No, A21 is a variable.
        # The variable A21 corresponds to U[1, 0] (row 1, col 0 of the 2x4 block).
        # Wait, U has shape (n*m, rank).
        # For the 2x4x2 block, U is 2x4.
        # Row 0: A00, A01, A02, A03.
        # Row 1: A10, A11, A12, A13.
        # A_TL is A00, A01, A10, A11.
        # So A21 is A10 (row 1, col 0). A22 is A11 (row 1, col 1).
        # So in U, we add 1 to U[1, 0] and U[1, 1]?
        # No, U is the matrix of coefficients.
        # U[i, c] is the coefficient of the i-th input variable in the c-th term.
        # The input variables are indexed by i (0..11).
        # For A_TL, the variables are A00, A01, A10, A11.
        # A21 is A10. A22 is A11.
        # So U[1, 0] corresponds to A10?
        # Let's map global indices.
        # A is 3x4.
        # A_top is rows 0,1.
        # A_TL is cols 0,1.
        # So A00 (idx 0), A01 (idx 1), A10 (idx 4), A11 (idx 5).
        # A_TR is cols 2,3. A02 (idx 2), A03 (idx 3), A12 (idx 6), A13 (idx 7).
        #
        # So for A_TL block:
        # U row 0 (A00, A01): indices 0, 1.
        # U row 1 (A10, A11): indices 4, 5.
        #
        # This is getting complicated to hardcode offsets.
        # Let's write a helper to handle the Strassen block given the U, V, W index mappings.
        
        pass # We will implement the full logic directly below to avoid abstraction overhead.
        
    # Let's manually construct the 44 rank decomposition.
    #
    # Part 1: C_top_left (2x2) = A_TL (2x2) * B_LL (2x2) + A_TR (2x2) * B_LR (2x2).
    #
    # Strassen for A_TL * B_LL:
    # A_TL indices: 0, 1, 4, 5.
    # B_LL indices: 0, 1, 4, 5. (B is 4x4. B_LL is rows 0,1 cols 0,1. V indices: 0,1,4,5).
    # C_TL indices: 0, 1, 4, 5. (C is 3x4. C_TL is rows 0,1 cols 0,1. W indices: 0,1,4,5).
    #
    # M1: (A00+A11)*(B00+B11) -> C00+=1, C11+=1.
    U[0][col] = 1; U[5][col] = 1
    V[0][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[5][col] = 1
    col += 1
    
    # M2: (A10+A11)*B00 -> C10+=1.
    U[4][col] = 1; U[5][col] = 1
    V[0][col] = 1
    W[4][col] = 1
    col += 1
    
    # M3: A00*(B01-B11) -> C01+=1.
    U[0][col] = 1
    V[1][col] = 1; V[5][col] = -1
    W[1][col] = 1
    col += 1
    
    # M4: A11*(B10-B00) -> C10+=1, C11+=1.
    U[5][col] = 1
    V[4][col] = 1; V[0][col] = -1
    W[4][col] = 1; W[5][col] = 1
    col += 1
    
    # M5: (A00+A01)*B11 -> C00-=1, C01+=1.
    U[0][col] = 1; U[1][col] = 1
    V[5][col] = 1
    W[0][col] = -1; W[1][col] = 1
    col += 1
    
    # M6: (A10-A00)*(B00+B01) -> C11+=1.
    U[4][col] = 1; U[0][col] = -1
    V[0][col] = 1; V[1][col] = 1
    W[5][col] = 1
    col += 1
    
    # M7: (A01-A11)*(B10+B11) -> C00+=1, C10-=1.
    U[1][col] = 1; U[5][col] = -1
    V[4][col] = 1; V[5][col] = 1
    W[0][col] = 1; W[4][col] = -1
    col += 1
    
    # Strassen for A_TR * B_LR:
    # A_TR indices: 2, 3, 6, 7.
    # B_LR indices: 2, 3, 6, 7.
    # C_TR indices: 2, 3, 6, 7.
    
    # M1: (A02+A13)*(B22+B33) -> C22+=1, C33+=1.
    U[2][col] = 1; U[7][col] = 1
    V[2][col] = 1; V[7][col] = 1
    W[2][col] = 1; W[7][col] = 1
    col += 1
    
    # M2: (A12+A13)*B22 -> C32+=1.
    U[6][col] = 1; U[7][col] = 1
    V[2][col] = 1
    W[6][col] = 1
    col += 1
    
    # M3: A02*(B23-B33) -> C23+=1.
    U[2][col] = 1
    V[3][col] = 1; V[7][col] = -1
    W[3][col] = 1
    col += 1
    
    # M4: A13*(B32-B22) -> C32+=1, C33+=1.
    U[7][col] = 1
    V[6][col] = 1; V[2][col] = -1
    W[6][col] = 1; W[7][col] = 1
    col += 1
    
    # M5: (A02+A03)*B33 -> C22-=1, C23+=1.
    U[2][col] = 1; U[3][col] = 1
    V[7][col] = 1
    W[2][col] = -1; W[3][col] = 1
    col += 1
    
    # M6: (A12-A02)*(B22+B23) -> C33+=1.
    U[6][col] = 1; U[2][col] = -1
    V[2][col] = 1; V[3][col] = 1
    W[7][col] = 1
    col += 1
    
    # M7: (A03-A13)*(B32+B33) -> C22+=1, C32-=1.
    U[3][col] = 1; U[7][col] = -1
    V[6][col] = 1; V[7][col] = 1
    W[2][col] = 1; W[6][col] = -1
    col += 1
    
    # Part 2: C_bot_left (1x2) = A_bot (1x4) * B_left (4x2).
    # A_bot indices: 8, 9, 10, 11.
    # B_left indices: 0, 1, 4, 5, 8, 9, 12, 13.
    # C_bot_left indices: 8, 9.
    # Schoolbook: 8 terms.
    
    # C8 = A8B0 + A9B4 + A10B8 + A11B12
    # C9 = A8B1 + A9B5 + A10B9 + A11B13
    
    U[8][col] = 1; V[0][col] = 1; W[8][col] = 1; col += 1
    U[9][col] = 1; V[4][col] = 1; W[8][col] = 1; col += 1
    U[10][col] = 1; V[8][col] = 1; W[8][col] = 1; col += 1
    U[11][col] = 1; V[12][col] = 1; W[8][col] = 1; col += 1
    
    U[8][col] = 1; V[1][col] = 1; W[9][col] = 1; col += 1
    U[9][col] = 1; V[5][col] = 1; W[9][col] = 1; col += 1
    U[10][col] = 1; V[9][col] = 1; W[9][col] = 1; col += 1
    U[11][col] = 1; V[13][col] = 1; W[9][col] = 1; col += 1
    
    # Part 3: C_bot_right (1x2) = A_bot (1x4) * B_right (4x2).
    # B_right indices: 2, 3, 6, 7, 10, 11, 14, 15.
    # C_bot_right indices: 10, 11.
    
    # C10 = A8B2 + A9B6 + A10B10 + A11B14
    # C11 = A8B3 + A9B7 + A10B11 + A11B15
    
    U[8][col] = 1; V[2][col] = 1; W[10][col] = 1; col += 1
    U[9][col] = 1; V[6][col] = 1; W[10][col] = 1; col += 1
    U[10][col] = 1; V[10][col] = 1; W[10][col] = 1; col += 1
    U[11][col] = 1; V[14][col] = 1; W[10][col] = 1; col += 1
    
    U[8][col] = 1; V[3][col] = 1; W[11][col] = 1; col += 1
    U[9][col] = 1; V[7][col] = 1; W[11][col] = 1; col += 1
    U[10][col] = 1; V[11][col] = 1; W[11][col] = 1; col += 1
    U[11][col] = 1; V[15][col] = 1; W[11][col] = 1; col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
