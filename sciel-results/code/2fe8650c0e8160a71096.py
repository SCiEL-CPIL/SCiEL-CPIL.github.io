import time

def construct():
    """
    Returns a rank-72 tensor decomposition for 4x4 * 4x9 matrix multiplication.
    Uses Strassen (rank 28) for two 4x4 blocks and Schoolbook (rank 16) for one 4x1 block.
    U: (4*4, 72)
    V: (4*9, 72)
    W: (9*4, 72)
    """
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 9
    rank = 72
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a schoolbook term for 2x2 * 2x2
    # a_idx: list of (global_idx, coeff) for A
    # b_idx: list of (global_idx, coeff) for B
    # c_idx: global_idx for C
    # col: column index in U,V,W
    def add_term(a_idx, b_idx, c_idx, col):
        for idx, coeff in a_idx:
            U[idx][col] += coeff
        for idx, coeff in b_idx:
            V[idx][col] += coeff
        W[c_idx][col] += 1

    # Define Strassen products for 4x4x4
    # Blocks:
    # A11: 0,1; 0,1 -> idx 0,1,4,5
    # A12: 0,1; 2,3 -> idx 2,3,6,7
    # A21: 2,3; 0,1 -> idx 8,9,12,13
    # A22: 2,3; 2,3 -> idx 10,11,14,15
    
    # B11: 0,1; 0,1 -> idx 0,1,4,5 (in B space, which is 4x4)
    # B12: 0,1; 2,3 -> idx 2,3,6,7
    # B21: 2,3; 0,1 -> idx 8,9,12,13
    # B22: 2,3; 2,3 -> idx 10,11,14,15
    
    # C11: 0,1; 0,1 -> idx 0,1,4,5 (in C space, 4x4)
    # C12: 0,1; 2,3 -> idx 2,3,6,7
    # C21: 2,3; 0,1 -> idx 8,9,12,13
    # C22: 2,3; 2,3 -> idx 10,11,14,15
    
    col = 0
    
    # M1 = (A11 + A22)(B11 + B22)
    # A' = A11 + A22
    # A'00 = A00 + A22 -> (0, 1), (10, 1)
    # A'01 = A01 + A23 -> (1, 1), (11, 1)
    # A'10 = A10 + A32 -> (4, 1), (14, 1)
    # A'11 = A11 + A33 -> (5, 1), (15, 1)
    
    # B' = B11 + B22
    # B'00 = B00 + B22 -> (0, 1), (10, 1)
    # B'01 = B01 + B23 -> (1, 1), (11, 1)
    # B'10 = B10 + B32 -> (4, 1), (14, 1)
    # B'11 = B11 + B33 -> (5, 1), (15, 1)
    
    # M100 -> C1100 (C00)
    add_term([(0,1),(10,1)], [(0,1),(10,1)], 0, col); col+=1
    # M101 -> C1200 (C02)
    add_term([(0,1),(10,1)], [(1,1),(11,1)], 2, col); col+=1
    # M110 -> C2100 (C20)
    add_term([(4,1),(14,1)], [(0,1),(10,1)], 8, col); col+=1
    # M111 -> C2200 (C22)
    add_term([(4,1),(14,1)], [(1,1),(11,1)], 10, col); col+=1
    
    # M2 = (A21 + A22)B11
    # A' = A21 + A22
    # A'00 = A20 + A22 -> (8, 1), (10, 1)
    # A'01 = A21 + A23 -> (9, 1), (11, 1)
    # A'10 = A30 + A32 -> (12, 1), (14, 1)
    # A'11 = A31 + A33 -> (13, 1), (15, 1)
    
    # B' = B11
    # B'00 = B00 -> (0, 1)
    # B'01 = B01 -> (1, 1)
    # B'10 = B10 -> (4, 1)
    # B'11 = B11 -> (5, 1)
    
    # M200 -> C2100 (C20)
    add_term([(8,1),(10,1)], [(0,1)], 8, col); col+=1
    # M201 -> C2101 (C21)
    add_term([(8,1),(10,1)], [(1,1)], 9, col); col+=1
    # M210 -> C2200 (C22)
    add_term([(12,1),(14,1)], [(0,1)], 10, col); col+=1
    # M211 -> C2201 (C23)
    add_term([(12,1),(14,1)], [(1,1)], 11, col); col+=1
    
    # M3 = A11(B12 - B22)
    # A' = A11
    # A'00 = A00 -> (0, 1)
    # A'01 = A01 -> (1, 1)
    # A'10 = A10 -> (4, 1)
    # A'11 = A11 -> (5, 1)
    
    # B' = B12 - B22
    # B'00 = B02 - B22 -> (2, 1), (10, -1)
    # B'01 = B03 - B23 -> (3, 1), (11, -1)
    # B'10 = B12 - B32 -> (6, 1), (14, -1)
    # B'11 = B13 - B33 -> (7, 1), (15, -1)
    
    # M300 -> C1200 (C02)
    add_term([(0,1)], [(2,1),(10,-1)], 2, col); col+=1
    # M301 -> C1201 (C03)
    add_term([(0,1)], [(3,1),(11,-1)], 3, col); col+=1
    # M310 -> C1210 (C12)
    add_term([(4,1)], [(2,1),(10,-1)], 6, col); col+=1
    # M311 -> C1211 (C13)
    add_term([(4,1)], [(3,1),(11,-1)], 7, col); col+=1
    
    # M4 = A22(B21 - B11)
    # A' = A22
    # A'00 = A22 -> (10, 1)
    # A'01 = A23 -> (11, 1)
    # A'10 = A32 -> (14, 1)
    # A'11 = A33 -> (15, 1)
    
    # B' = B21 - B11
    # B'00 = B20 - B00 -> (8, 1), (0, -1)
    # B'01 = B21 - B01 -> (9, 1), (1, -1)
    # B'10 = B30 - B10 -> (12, 1), (4, -1)
    # B'11 = B31 - B11 -> (13, 1), (5, -1)
    
    # M400 -> C1100 (C00)
    add_term([(10,1)], [(8,1),(0,-1)], 0, col); col+=1
    # M401 -> C1101 (C01)
    add_term([(10,1)], [(9,1),(1,-1)], 1, col); col+=1
    # M410 -> C1110 (C10)
    add_term([(14,1)], [(8,1),(0,-1)], 4, col); col+=1
    # M411 -> C1111 (C11)
    add_term([(14,1)], [(9,1),(1,-1)], 5, col); col+=1
    
    # M5 = (A11 + A12)B22
    # A' = A11 + A12
    # A'00 = A00 + A02 -> (0, 1), (2, 1)
    # A'01 = A01 + A03 -> (1, 1), (3, 1)
    # A'10 = A10 + A12 -> (4, 1), (6, 1)
    # A'11 = A11 + A13 -> (5, 1), (7, 1)
    
    # B' = B22
    # B'00 = B22 -> (10, 1)
    # B'01 = B23 -> (11, 1)
    # B'10 = B32 -> (14, 1)
    # B'11 = B33 -> (15, 1)
    
    # M500 -> C1100 (C00)
    add_term([(0,1),(2,1)], [(10,1)], 0, col); col+=1
    # M501 -> C1200 (C02)
    add_term([(0,1),(2,1)], [(11,1)], 2, col); col+=1
    # M510 -> C1110 (C10)
    add_term([(4,1),(6,1)], [(10,1)], 4, col); col+=1
    # M511 -> C1210 (C12)
    add_term([(4,1),(6,1)], [(11,1)], 6, col); col+=1
    
    # M6 = (A21 - A11)(B11 + B12)
    # A' = A21 - A11
    # A'00 = A20 - A00 -> (8, 1), (0, -1)
    # A'01 = A21 - A01 -> (9, 1), (1, -1)
    # A'10 = A30 - A10 -> (12, 1), (4, -1)
    # A'11 = A31 - A11 -> (13, 1), (5, -1)
    
    # B' = B11 + B12
    # B'00 = B00 + B02 -> (0, 1), (2, 1)
    # B'01 = B01 + B03 -> (1, 1), (3, 1)
    # B'10 = B10 + B12 -> (4, 1), (6, 1)
    # B'11 = B11 + B13 -> (5, 1), (7, 1)
    
    # M600 -> C1100 (C00)
    add_term([(8,1),(0,-1)], [(0,1),(2,1)], 0, col); col+=1
    # M601 -> C1101 (C01)
    add_term([(8,1),(0,-1)], [(1,1),(3,1)], 1, col); col+=1
    # M610 -> C1110 (C10)
    add_term([(12,1),(4,-1)], [(0,1),(2,1)], 4, col); col+=1
    # M611 -> C1111 (C11)
    add_term([(12,1),(4,-1)], [(1,1),(3,1)], 5, col); col+=1
    
    # M7 = (A12 - A22)(B21 + B22)
    # A' = A12 - A22
    # A'00 = A02 - A22 -> (2, 1), (10, -1)
    # A'01 = A03 - A23 -> (3, 1), (11, -1)
    # A'10 = A12 - A32 -> (6, 1), (14, -1)
    # A'11 = A13 - A33 -> (7, 1), (15, -1)
    
    # B' = B21 + B22
    # B'00 = B20 + B22 -> (8, 1), (10, 1)
    # B'01 = B21 + B23 -> (9, 1), (11, 1)
    # B'10 = B30 + B32 -> (12, 1), (14, 1)
    # B'11 = B31 + B33 -> (13, 1), (15, 1)
    
    # M700 -> C1100 (C00)
    add_term([(2,1),(10,-1)], [(8,1),(10,1)], 0, col); col+=1
    # M701 -> C1101 (C01)
    add_term([(2,1),(10,-1)], [(9,1),(11,1)], 1, col); col+=1
    # M710 -> C1110 (C10)
    add_term([(6,1),(14,-1)], [(8,1),(10,1)], 4, col); col+=1
    # M711 -> C1111 (C11)
    add_term([(6,1),(14,-1)], [(9,1),(11,1)], 5, col); col+=1
    
    # col should be 28 now.
    
    # Now for the second 4x4 block (cols 4-7 of B, 4-7 of C)
    # We repeat the same U, but shift V and W indices.
    # V indices for B block 2: 16..31
    # W indices for C block 2: 16..31
    
    # We can just copy the U, V, W patterns from the first block.
    # But we need to be careful with the indices in add_term.
    # It's easier to re-run the logic with offset.
    
    # Let's just copy the values.
    # U[:, 28:56] = U[:, 0:28]
    for i in range(16):
        for r in range(28):
            U[i][28+r] = U[i][r]
            
    # V[16:32, 28:56] = V[0:16, 0:28]
    for i in range(16):
        for r in range(28):
            V[16+i][28+r] = V[i][r]
            
    # W[16:32, 28:56] = W[0:16, 0:28]
    for i in range(16):
        for r in range(28):
            W[16+i][28+r] = W[i][r]
            
    # col is now 56.
    
    # Now for the 4x1 block (col 8 of B, col 8 of C)
    # Schoolbook 4x4x1
    # A is 4x4 (16 entries)
    # B is 4x1 (4 entries, indices 32..35)
    # C is 4x1 (4 entries, indices 32..35 in W? No, W is 9x4 = 36.
    # C index k*n + i. k=8, i=0..3. Indices 32..35.
    
    for i in range(4):
        for j in range(4):
            U[i*4+j][56 + i*4+j] = 1
            V[32+j][56 + i*4+j] = 1
            W[32+i][56 + i*4+j] = 1
            
    # Total rank 72.
    
    return {"U": U, "V": V, "W": W}
