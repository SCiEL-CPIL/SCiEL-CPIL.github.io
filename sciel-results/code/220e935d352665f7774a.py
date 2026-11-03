"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-42 decomposition using Strassen's algorithm for 2x2x2 blocks.
    
    A (2x4) = [A_L | A_R] where A_L, A_R are 2x2
    B (4x6) = [B_T; B_B] where B_T, B_B are 2x6
    C (2x6) = A_L * B_T + A_R * B_B
    
    Each 2x2 * 2x6 is split into three 2x2 * 2x2 products, each using Strassen (rank 7).
    Total rank = 2 * 3 * 7 = 42.
    """
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen coefficients for 2x2x2
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    
    # For each Strassen term, the U coefficients (for A elements a,b,c,d)
    # and V coefficients (for B elements e,f,g,h)
    strassen_U = [
        [1, 0, 0, 1],   # M1: a+d
        [0, 0, 1, 1],   # M2: c+d
        [1, 0, 0, 0],   # M3: a
        [0, 0, 0, 1],   # M4: d
        [1, 1, 0, 0],   # M5: a+b
        [0, 0, 1, -1],  # M6: c-a
        [0, 1, 0, -1],  # M7: b-d
    ]
    strassen_V = [
        [1, 0, 0, 1],   # M1: e+h
        [1, 0, 0, 0],   # M2: e
        [0, 1, 0, -1],  # M3: f-h
        [0, 0, 1, -1],  # M4: g-e
        [0, 0, 0, 1],   # M5: h
        [1, 1, 0, 0],   # M6: e+f
        [0, 0, 1, 1],   # M7: g+h
    ]
    strassen_W = [
        [1, 0, 0, 1],   # M1: C11, C22
        [0, 0, 0, 0],   # M2: 
        [0, 1, 0, 0],   # M3: C12
        [1, 0, 1, 0],   # M4: C11, C21
        [0, 1, 0, 0],   # M5: C12
        [0, 0, 0, 1],   # M6: C22
        [1, 0, 0, 0],   # M7: C11
    ]
    # Fix W coefficients:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    strassen_W = [
        [1, 0, 0, 1],   # M1: C11 += 1, C22 += 1
        [0, 0, 1, -1],  # M2: C21 += 1, C22 += -1
        [0, 1, 0, 1],   # M3: C12 += 1, C22 += 1
        [1, 0, 1, 0],   # M4: C11 += 1, C21 += 1
        [-1, 1, 0, 0],  # M5: C11 += -1, C12 += 1
        [0, 0, 0, 1],   # M6: C22 += 1
        [1, 0, 0, 0],   # M7: C11 += 1
    ]
    
    # A element indices: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # In flat index for 2x2: a=0, b=1, c=2, d=3
    
    # B element indices for 2x2: e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    # In flat index for 2x2: e=0, f=1, g=2, h=3
    
    # C element indices for 2x2: C11=C[0,0], C12=C[0,1], C21=C[1,0], C22=C[1,1]
    # In flat index for 2x2: C11=0, C12=1, C21=2, C22=3
    
    # Now build the full decomposition
    # A_L is A[0:2, 0:2], A_R is A[0:2, 2:4]
    # B_T is B[0:2, 0:6], B_B is B[2:4, 0:6]
    
    # For A_L * B_T:
    # B_T = [BT1 | BT2 | BT3] where BTj is 2x2, j=0,1,2
    # BT1 = B[0:2, 0:2], BT2 = B[0:2, 2:4], BT3 = B[0:2, 4:6]
    
    # For A_R * B_B:
    # B_B = [BB1 | BB2 | BB3] where BBj is 2x2, j=0,1,2
    # BB1 = B[2:4, 0:2], BB2 = B[2:4, 2:4], BB3 = B[2:4, 4:6]
    
    # Output C = [C1 C2 C3] where Cj is 2x2
    # C1 = A_L*BT1 + A_R*BB1
    # C2 = A_L*BT2 + A_R*BB2
    # C3 = A_L*BT3 + A_R*BB3
    
    col = 0
    # Process A_L * B_T (3 blocks)
    for block in range(3):
        for s in range(7):  # 7 Strassen terms
            # U: A_L elements
            # A_L flat: a=0, b=1, c=2, d=3 in the 2x2 submatrix
            # In full A (2x4) flat: A_L is rows 0-1, cols 0-1
            # Flat index: row*4 + col
            # a = A[0,0] = 0, b = A[0,1] = 1, c = A[1,0] = 4, d = A[1,1] = 5
            A_L_flat = [0, 1, 4, 5]  # a, b, c, d
            
            # V: B_T block elements
            # B_T is B[0:2, 0:6], block is cols block*2 to block*2+1
            # e = B[0, block*2], f = B[0, block*2+1], g = B[1, block*2], h = B[1, block*2+1]
            # In full B (4x6) flat: row*6 + col
            # e = 0*6 + block*2 = block*2
            # f = 0*6 + block*2+1 = block*2+1
            # g = 1*6 + block*2 = 6 + block*2
            # h = 1*6 + block*2+1 = 6 + block*2+1
            B_T_block_flat = [block*2, block*2+1, 6+block*2, 6+block*2+1]  # e, f, g, h
            
            # W: C block elements
            # C block is C[0:2, block*2:block*2+2]
            # C11 = C[0, block*2], C12 = C[0, block*2+1], C21 = C[1, block*2], C22 = C[1, block*2+1]
            # In full C (2x6) flat: row*6 + col
            # C11 = 0*6 + block*2 = block*2
            # C12 = 0*6 + block*2+1 = block*2+1
            # C21 = 1*6 + block*2 = 6 + block*2
            # C22 = 1*6 + block*2+1 = 6 + block*2+1
            C_block_flat = [block*2, block*2+1, 6+block*2, 6+block*2+1]  # C11, C12, C21, C22
            
            for idx in range(4):
                U[A_L_flat[idx]][col] += strassen_U[s][idx]
                V[B_T_block_flat[idx]][col] += strassen_V[s][idx]
                W[C_block_flat[idx]][col] += strassen_W[s][idx]
            col += 1
    
    # Process A_R * B_B (3 blocks)
    for block in range(3):
        for s in range(7):  # 7 Strassen terms
            # U: A_R elements
            # A_R is A[0:2, 2:4]
            # a = A[0,2] = 2, b = A[0,3] = 3, c = A[1,2] = 6, d = A[1,3] = 7
            A_R_flat = [2, 3, 6, 7]  # a, b, c, d
            
            # V: B_B block elements
            # B_B is B[2:4, 0:6], block is cols block*2 to block*2+1
            # e = B[2, block*2], f = B[2, block*2+1], g = B[3, block*2], h = B[3, block*2+1]
            # In full B (4x6) flat: row*6 + col
            # e = 2*6 + block*2 = 12 + block*2
            # f = 2*6 + block*2+1 = 12 + block*2+1
            # g = 3*6 + block*2 = 18 + block*2
            # h = 3*6 + block*2+1 = 18 + block*2+1
            B_B_block_flat = [12+block*2, 12+block*2+1, 18+block*2, 18+block*2+1]  # e, f, g, h
            
            # W: C block elements (same as before, since C is shared)
            C_block_flat = [block*2, block*2+1, 6+block*2, 6+block*2+1]  # C11, C12, C21, C22
            
            for idx in range(4):
                U[A_R_flat[idx]][col] += strassen_U[s][idx]
                V[B_B_block_flat[idx]][col] += strassen_V[s][idx]
                W[C_block_flat[idx]][col] += strassen_W[s][idx]
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
