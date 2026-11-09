"""Rewritten program for 4x4 * 4x9 matrix multiplication tensor decomposition."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 9
    
    # Strassen 2x2x2x2 decomposition: 7 terms
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # C00 = M1+M4-M5+M7, C01 = M3+M5, C10 = M2+M4, C11 = M1-M2+M3+M6
    
    # U coefficients for a,b,c,d (index 0,1,2,3)
    strassen_U = [
        [1, 0, 0, 1],  # M1: a+d
        [0, 0, 1, 1],  # M2: c+d
        [1, 0, 0, 0],  # M3: a
        [0, 0, 0, 1],  # M4: d
        [1, 1, 0, 0],  # M5: a+b
        [-1, 0, 1, 0], # M6: c-a
        [0, 1, 0, -1], # M7: b-d
    ]
    
    # V coefficients for e,f,g,h (index 0,1,2,3)
    strassen_V = [
        [1, 0, 0, 1],  # M1: e+h
        [1, 0, 0, 0],  # M2: e
        [0, 1, 0, -1], # M3: f-h
        [0, 0, 1, -1], # M4: g-e
        [0, 0, 0, 1],  # M5: h
        [1, 1, 0, 0],  # M6: e+f
        [0, 0, 1, 1],  # M7: g+h
    ]
    
    # W coefficients for C00,C01,C10,C11 (index 0,1,2,3)
    strassen_W = [
        [1, 0, 0, 1],   # M1 contributes to C00 and C11
        [0, 0, 1, -1],  # M2 contributes to C10 and C11
        [0, 1, 0, 1],   # M3 contributes to C01 and C11
        [1, 0, 1, 0],   # M4 contributes to C00 and C10
        [-1, 1, 0, 0],  # M5 contributes to C00 and C01
        [0, 0, 0, 1],   # M6 contributes to C11
        [1, 0, 0, 0],   # M7 contributes to C00
    ]
    
    # Build 4x4x4 Strassen decomposition using tensor product of 2x2 Strassen
    # A is 4x4, split into 2x2 blocks: A11, A12, A21, A22
    # B is 4x4, split into 2x2 blocks: B11, B12, B21, B22
    # 
    # Level 1 Strassen (4x4):
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Each Mi is a product of two 2x2 matrices, so we apply Strassen 2x2 to each.
    # Total: 7 * 7 = 49 terms.
    
    # U for 4x4 matrix A (16 entries: a00,a01,a10,a11,a20,a21,a30,a31,a40,a41,a50,a51,a60,a61,a70,a71)
    # Actually let's use row-major: A[i,j] where i,j in 0..3
    # Index: i*4+j
    
    # V for 4x4 matrix B (16 entries)
    # Index: i*4+j
    
    # W for 4x4 matrix C (16 entries)
    # Index: i*4+j
    
    # For each of the 7 level-1 Strassen terms, we have:
    # - A linear combination of 2x2 blocks of A
    # - A linear combination of 2x2 blocks of B
    # - A linear combination of 2x2 blocks of C
    
    # Let's define the level-1 Strassen coefficients:
    # For each term t in 0..6:
    #   U1[t] = coefficients for A blocks (A11, A12, A21, A22)
    #   V1[t] = coefficients for B blocks (B11, B12, B21, B22)
    #   W1[t] = coefficients for C blocks (C11, C12, C21, C22)
    
    # A blocks: A11 = [[A00,A01],[A10,A11]], A12 = [[A02,A03],[A12,A13]]
    #           A21 = [[A20,A21],[A30,A31]], A22 = [[A22,A23],[A32,A33]]
    # B blocks: B11 = [[B00,B01],[B10,B11]], B12 = [[B02,B03],[B12,B13]]
    #           B21 = [[B20,B21],[B30,B31]], B22 = [[B22,B23],[B32,B33]]
    # C blocks: C11 = [[C00,C01],[C10,C11]], C12 = [[C02,C03],[C12,C13]]
    #           C21 = [[C20,C21],[C30,C31]], C22 = [[C22,C23],[C32,C33]]
    
    # Level 1 Strassen:
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    
    # U1 coefficients (for A11, A12, A21, A22):
    U1 = [
        [1, 0, 0, 1],  # M1: A11 + A22
        [0, 0, 1, 1],  # M2: A21 + A22
        [1, 0, 0, 0],  # M3: A11
        [0, 0, 0, 1],  # M4: A22
        [1, 1, 0, 0],  # M5: A11 + A12
        [-1, 0, 1, 0], # M6: A21 - A11
        [0, 1, 0, -1], # M7: A12 - A22
    ]
    
    # V1 coefficients (for B11, B12, B21, B22):
    V1 = [
        [1, 0, 0, 1],  # M1: B11 + B22
        [1, 0, 0, 0],  # M2: B11
        [0, 1, 0, -1], # M3: B12 - B22
        [0, 0, 1, -1], # M4: B21 - B11
        [0, 0, 0, 1],  # M5: B22
        [1, 1, 0, 0],  # M6: B11 + B12
        [0, 0, 1, 1],  # M7: B21 + B22
    ]
    
    # W1 coefficients (for C11, C12, C21, C22):
    W1 = [
        [1, 0, 0, 1],   # M1: C11 + C22
        [0, 0, 1, -1],  # M2: C21 - C22
        [0, 1, 0, 1],   # M3: C12 + C22
        [1, 0, 1, 0],   # M4: C11 + C21
        [-1, 1, 0, 0],  # M5: -C11 + C12
        [0, 0, 0, 1],   # M6: C22
        [1, 0, 0, 0],   # M7: C11
    ]
    
    # Now, for each level-1 term t, and each level-2 Strassen term s (0..6):
    # The scalar product is:
    #   (sum of A entries with coefficients from U1[t] and strassen_U[s])
    #   (sum of B entries with coefficients from V1[t] and strassen_V[s])
    #   (sum of C entries with coefficients from W1[t] and strassen_W[s])
    
    # Let's build the U, V, W matrices for the 4x4x4 part.
    # U has 16 rows (for A entries), V has 16 rows (for B entries), W has 16 rows (for C entries).
    # Each term adds one column.
    
    rank_444 = 49
    U_444 = [[0] * rank_444 for _ in range(16)]
    V_444 = [[0] * rank_444 for _ in range(16)]
    W_444 = [[0] * rank_444 for _ in range(16)]
    
    col = 0
    for t in range(7):  # level 1 Strassen term
        for s in range(7):  # level 2 Strassen term
            # A coefficients: combine U1[t] (block coefficients) with strassen_U[s] (entry coefficients within block)
            # A11 entries: indices 0,1,4,5 (A00,A01,A10,A11)
            # A12 entries: indices 2,3,6,7 (A02,A03,A12,A13)
            # A21 entries: indices 8,9,12,13 (A20,A21,A30,A31)
            # A22 entries: indices 10,11,14,15 (A22,A23,A32,A33)
            
            # Block to entry mapping:
            # Block 0 (A11): entries 0,1,4,5 -> strassen_U[s][0], strassen_U[s][1], strassen_U[s][2], strassen_U[s][3]
            # Block 1 (A12): entries 2,3,6,7
            # Block 2 (A21): entries 8,9,12,13
            # Block 3 (A22): entries 10,11,14,15
            
            block_entries = [
                [0, 1, 4, 5],   # A11
                [2, 3, 6, 7],   # A12
                [8, 9, 12, 13], # A21
                [10, 11, 14, 15] # A22
            ]
            
            for b in range(4):
                coeff_block = U1[t][b]
                if coeff_block == 0:
                    continue
                for e_idx in range(4):
                    entry = block_entries[b][e_idx]
                    coeff_entry = strassen_U[s][e_idx]
                    if coeff_entry != 0:
                        U_444[entry][col] += coeff_block * coeff_entry
            
            # B coefficients: combine V1[t] (block coefficients) with strassen_V[s] (entry coefficients within block)
            # B11 entries: indices 0,1,4,5 (B00,B01,B10,B11)
            # B12 entries: indices 2,3,6,7 (B02,B03,B12,B13)
            # B21 entries: indices 8,9,12,13 (B20,B21,B30,B31)
            # B22 entries: indices 10,11,14,15 (B22,B23,B32,B33)
            
            block_entries_B = [
                [0, 1, 4, 5],   # B11
                [2, 3, 6, 7],   # B12
                [8, 9, 12, 13], # B21
                [10, 11, 14, 15] # B22
            ]
            
            for b in range(4):
                coeff_block = V1[t][b]
                if coeff_block == 0:
                    continue
                for e_idx in range(4):
                    entry = block_entries_B[b][e_idx]
                    coeff_entry = strassen_V[s][e_idx]
                    if coeff_entry != 0:
                        V_444[entry][col] += coeff_block * coeff_entry
            
            # C coefficients: combine W1[t] (block coefficients) with strassen_W[s] (entry coefficients within block)
            # C11 entries: indices 0,1,4,5 (C00,C01,C10,C11)
            # C12 entries: indices 2,3,6,7 (C02,C03,C12,C13)
            # C21 entries: indices 8,9,12,13 (C20,C21,C30,C31)
            # C22 entries: indices 10,11,14,15 (C22,C23,C32,C33)
            
            block_entries_C = [
                [0, 1, 4, 5],   # C11
                [2, 3, 6, 7],   # C12
                [8, 9, 12, 13], # C21
                [10, 11, 14, 15] # C22
            ]
            
            for b in range(4):
                coeff_block = W1[t][b]
                if coeff_block == 0:
                    continue
                for e_idx in range(4):
                    entry = block_entries_C[b][e_idx]
                    coeff_entry = strassen_W[s][e_idx]
                    if coeff_entry != 0:
                        W_444[entry][col] += coeff_block * coeff_entry
            
            col += 1
    
    # Now build the schoolbook decomposition for 4x4 * 4x5 (last 5 columns of B)
    # B has 9 columns, we've handled columns 0-3 (first 4 columns) with Strassen.
    # We need to handle columns 4-8 (5 columns) with schoolbook.
    # For each (i, j, k) where i in 0..3, j in 0..3, k in 4..8:
    #   U[i*4+j] = 1
    #   V[j*4+k] = 1
    #   W[k*4+i] = 1
    
    rank_445 = 4 * 4 * 5  # 80
    U_445 = [[0] * rank_445 for _ in range(16)]
    V_445 = [[0] * rank_445 for _ in range(36)]  # 4*9 = 36 entries for B
    W_445 = [[0] * rank_445 for _ in range(36)]  # 4*9 = 36 entries for C
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                U_445[i * 4 + j][col] = 1
                V[j * 9 + k][col] = 1
                W[k * 4 + i][col] = 1
                col += 1
    
    # Combine: total rank = 49 + 80 = 129
    total_rank = rank_444 + rank_445
    
    # Build final U, V, W matrices
    # U: 16 x total_rank
    # V: 36 x total_rank
    # W: 36 x total_rank
    
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(36)]
    W = [[0] * total_rank for _ in range(36)]
    
    # Copy Strassen 4x4x4 part (first 49 columns)
    for r in range(16):
        for c in range(rank_444):
            U[r][c] = U_444[r][c]
            V[r][c] = V_444[r][c]
            W[r][c] = W_444[r][c]
    
    # Copy schoolbook 4x4x5 part (next 80 columns)
    for r in range(16):
        for c in range(rank_445):
            U[r][rank_444 + c] = U_445[r][c]
    for r in range(36):
        for c in range(rank_445):
            V[r][rank_444 + c] = V_445[r][c]
            W[r][rank_444 + c] = W_445[r][c]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
