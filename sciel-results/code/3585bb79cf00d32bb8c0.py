import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 49 for 4x4x4 + Rank 16 for 4x4x1
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 + rank_441
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Part 1: 4x4x4 (first 4 columns of B) ---
    # We need the rank-7 decomposition for 2x2x2
    # U22: (4, 7), V22: (4, 7), W22: (4, 7)
    U22 = [[0]*7 for _ in range(4)]
    V22 = [[0]*7 for _ in range(4)]
    W22 = [[0]*7 for _ in range(4)]
    
    # Strassen 2x2
    # M0 = (A00+A11)(B00+B11)
    U22[0][0] = 1; U22[3][0] = 1
    V22[0][0] = 1; V22[3][0] = 1
    W22[0][0] = 1; W22[3][0] = 1 # C00, C11
    
    # M1 = (A10+A11)B00
    U22[2][1] = 1; U22[3][1] = 1
    V22[0][1] = 1
    W22[1][1] = 1; W22[3][1] = -1 # C10, C11
    
    # M2 = A00(B01-B11)
    U22[0][2] = 1
    V22[1][2] = 1; V22[3][2] = -1
    W22[2][2] = 1 # C01
    
    # M3 = A11(B10-B00)
    U22[3][3] = 1
    V22[2][3] = 1; V22[0][3] = -1
    W22[0][3] = 1; W22[1][3] = 1 # C00, C10
    
    # M4 = (A00+A01)B11
    U22[0][4] = 1; U22[1][4] = 1
    V22[3][4] = 1
    W22[2][4] = 1; W22[0][4] = -1 # C01, C00
    
    # M5 = (A10-A00)(B00+B01)
    U22[2][5] = 1; U22[0][5] = -1
    V22[0][5] = 1; V22[1][5] = 1
    W22[3][5] = 1 # C11
    
    # M6 = (A01-A11)(B10+B11)
    U22[1][6] = 1; U22[3][6] = -1
    V22[2][6] = 1; V22[3][6] = 1
    W22[0][6] = 1 # C00

    # Now build 4x4x4 using 2x2x2 blocks
    # A blocks: A00 (0,0), A01 (0,1), A10 (1,0), A11 (1,1) in 2x2 grid of 2x2 matrices
    # Global index for A[i,j] is i*4+j.
    # Block A00 covers i in 0,1 and j in 0,1. Global indices: 0,1,4,5.
    # Block A01 covers i in 0,1 and j in 2,3. Global indices: 2,3,6,7.
    # Block A10 covers i in 2,3 and j in 0,1. Global indices: 8,9,12,13.
    # Block A11 covers i in 2,3 and j in 2,3. Global indices: 10,11,14,15.
    
    # B blocks: B00 (0,0), B01 (0,1), B10 (1,0), B11 (1,1)
    # Global index for B[j,k] is j*4+k.
    # Block B00 covers j in 0,1 and k in 0,1. Global indices: 0,1,4,5.
    # Block B01 covers j in 0,1 and k in 2,3. Global indices: 2,3,6,7.
    # Block B10 covers j in 2,3 and k in 0,1. Global indices: 8,9,12,13.
    # Block B11 covers j in 2,3 and k in 2,3. Global indices: 10,11,14,15.
    
    # C blocks: C00 (0,0), C01 (0,1), C10 (1,0), C11 (1,1)
    # Global index for C[i,k] is k*4+i (per W convention k*n+i).
    # Block C00 covers k in 0,1 and i in 0,1. Global indices: 0,1,4,5.
    # Block C01 covers k in 2,3 and i in 0,1. Global indices: 8,9,12,13.
    # Block C10 covers k in 0,1 and i in 2,3. Global indices: 2,3,6,7.
    # Block C11 covers k in 2,3 and i in 2,3. Global indices: 10,11,14,15.

    # Strassen 4x4 steps
    # k=0: X = A00+A11, Y = B00+B11, Z -> C00(+1), C11(+1)
    # k=1: X = A10+A11, Y = B00, Z -> C10(+1), C11(-1)
    # k=2: X = A00, Y = B01-B11, Z -> C01(+1)
    # k=3: X = A11, Y = B10-B00, Z -> C00(+1), C10(+1)
    # k=4: X = A00+A01, Y = B11, Z -> C01(+1), C00(-1)
    # k=5: X = A10-A00, Y = B00+B01, Z -> C11(+1)
    # k=6: X = A01-A11, Y = B10+B11, Z -> C00(+1)
    
    # Helper to add a block contribution
    def add_block_contribution(U, V, W, U22, V22, W22, 
                               x_blocks, y_blocks, z_blocks, 
                               a_block_offsets, b_block_offsets, c_block_offsets,
                               rank_offset):
        # x_blocks: list of (block_idx, coeff) for X
        # y_blocks: list of (block_idx, coeff) for Y
        # z_blocks: list of (block_idx, coeff) for Z
        
        # For each rank r22 in 0..6
        for r22 in range(7):
            r = rank_offset + r22
            # U part
            # U22[p,q,r22] is the coefficient for the local element (p,q) of X
            # X is formed from A blocks.
            # X_local[p,q] = sum_{block, coeff} coeff * A_block[p,q]
            # So U_global[i,j,r] = sum_{block, coeff} coeff * U22[p,q,r22] where (i,j) is in block.
            
            # Iterate over local indices p,q of the 2x2 block
            for p in range(2):
                for q in range(2):
                    val_u22 = U22[p*2+q][r22]
                    if val_u22 == 0: continue
                    
                    # Add to U for all A blocks contributing to X
                    for b_idx, coeff in x_blocks:
                        # Get global offsets for this block
                        # b_idx 0: A00, 1: A01, 2: A10, 3: A11
                        # Offsets for A00: [0,1,4,5] corresponding to (0,0),(0,1),(1,0),(1,1)
                        # We need to map local (p,q) to global index.
                        # Block 0 (A00): global_i = p, global_j = q. idx = p*4+q.
                        # Block 1 (A01): global_i = p, global_j = 2+q. idx = p*4+(2+q).
                        # Block 2 (A10): global_i = 2+p, global_j = q. idx = (2+p)*4+q.
                        # Block 3 (A11): global_i = 2+p, global_j = 2+q. idx = (2+p)*4+(2+q).
                        
                        if b_idx == 0:
                            gi = p; gj = q
                        elif b_idx == 1:
                            gi = p; gj = 2+q
                        elif b_idx == 2:
                            gi = 2+p; gj = q
                        else:
                            gi = 2+p; gj = 2+q
                            
                        idx_u = gi * 4 + gj
                        U[idx_u][r] += coeff * val_u22

            # V part
            for p in range(2): # local row j
                for q in range(2): # local col k
                    val_v22 = V22[p*2+q][r22]
                    if val_v22 == 0: continue
                    
                    for b_idx, coeff in y_blocks:
                        if b_idx == 0: # B00
                            gj = p; gk = q
                        elif b_idx == 1: # B01
                            gj = p; gk = 2+q
                        elif b_idx == 2: # B10
                            gj = 2+p; gk = q
                        else: # B11
                            gj = 2+p; gk = 2+q
                            
                        idx_v = gj * 4 + gk
                        V[idx_v][r] += coeff * val_v22

            # W part
            for p in range(2): # local row k
                for q in range(2): # local col i
                    val_w22 = W22[p*2+q][r22]
                    if val_w22 == 0: continue
                    
                    for b_idx, coeff in z_blocks:
                        if b_idx == 0: # C00
                            gk = p; gi = q
                        elif b_idx == 1: # C01
                            gk = 2+p; gi = q
                        elif b_idx == 2: # C10
                            gk = p; gi = 2+q
                        else: # C11
                            gk = 2+p; gi = 2+q
                            
                        idx_w = gk * 4 + gi
                        W[idx_w][r] += coeff * val_w22

    # Define the 7 steps
    # b_idx: 0=A00/B00/C00, 1=A01/B01/C01, 2=A10/B10/C10, 3=A11/B11/C11
    
    # k=0
    x0 = [(0, 1), (3, 1)]
    y0 = [(0, 1), (3, 1)]
    z0 = [(0, 1), (3, 1)]
    add_block_contribution(U, V, W, U22, V22, W22, x0, y0, z0, None, None, None, 0)
    
    # k=1
    x1 = [(2, 1), (3, 1)]
    y1 = [(0, 1)]
    z1 = [(2, 1), (3, -1)]
    add_block_contribution(U, V, W, U22, V22, W22, x1, y1, z1, None, None, None, 7)
    
    # k=2
    x2 = [(0, 1)]
    y2 = [(1, 1), (3, -1)]
    z2 = [(1, 1)]
    add_block_contribution(U, V, W, U22, V22, W22, x2, y2, z2, None, None, None, 14)
    
    # k=3
    x3 = [(3, 1)]
    y3 = [(2, 1), (0, -1)]
    z3 = [(0, 1), (2, 1)]
    add_block_contribution(U, V, W, U22, V22, W22, x3, y3, z3, None, None, None, 21)
    
    # k=4
    x4 = [(0, 1), (1, 1)]
    y4 = [(3, 1)]
    z4 = [(1, 1), (0, -1)]
    add_block_contribution(U, V, W, U22, V22, W22, x4, y4, z4, None, None, None, 28)
    
    # k=5
    x5 = [(2, 1), (0, -1)]
    y5 = [(0, 1), (1, 1)]
    z5 = [(3, 1)]
    add_block_contribution(U, V, W, U22, V22, W22, x5, y5, z5, None, None, None, 35)
    
    # k=6
    x6 = [(1, 1), (3, -1)]
    y6 = [(2, 1), (3, 1)]
    z6 = [(0, 1)]
    add_block_contribution(U, V, W, U22, V22, W22, x6, y6, z6, None, None, None, 42)
    
    # --- Part 2: 4x4x1 (last 1 column of B) ---
    # Schoolbook rank 16
    # A is 4x4, B_col is 4x1 (column 4 of B, index j*5+4)
    # C_col is 4x1 (column 4 of C, index k=4, so W index 4*4+i = 16+i)
    for i in range(4):
        for j in range(4):
            r = 49 + i*4 + j
            idx_u = i*4 + j
            idx_v = j*5 + 4
            idx_w = 4*4 + i
            U[idx_u][r] = 1
            V[idx_v][r] = 1
            W[idx_w][r] = 1
            
    return {"U": U, "V": V, "W": W}
