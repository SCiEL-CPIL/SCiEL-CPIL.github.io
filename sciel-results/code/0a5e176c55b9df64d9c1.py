import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=9
    # U: 16 x R, V: 36 x R, W: 36 x R
    
    # 1. Decomposition for 4x4x4 (rank 49)
    # We need U4 (16x49), V4 (16x49), W4 (16x49)
    # Note: For 4x4x4, V index is j*4+k. Since m=4, p_part=4.
    # But in the full problem, m=4, p=9.
    # The V matrix for the full problem has rows j*9+k.
    # For the first 4 columns of B (k=0..3), the rows are j*9+k.
    # For the 4x4x4 part, we use k=0..3.
    
    # Let's generate U4, V4, W4 for 4x4x4
    # U4 rows: i*4+j (i,j in 0..3)
    # V4 rows: j*4+k (j,k in 0..3) -> In full V, this maps to j*9+k
    # W4 rows: k*4+i (k,i in 0..3) -> In full W, this maps to k*4+i (since p=9, but W index is k*n+i)
    # Wait, W index is k*n+i. For n=4, k=0..3, i=0..3.
    # So W4 rows are k*4+i. This matches the full W index for k=0..3.
    
    U4 = np.zeros((16, 49))
    V4 = np.zeros((16, 49))
    W4 = np.zeros((16, 49))
    
    # Strassen 2x2x2 matrices
    U2 = np.array([
        [1, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, -1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ]) # Rows: (0,0), (0,1), (1,0), (1,1)
    
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ])
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Compose for 4x4x4
    # Outer Strassen terms
    # A blocks: A00, A01, A10, A11
    # B blocks: B00, B10, B11, B01? No, B is 4x4.
    # B blocks: B00 (top-left), B01 (top-right), B10 (bot-left), B11 (bot-right)
    
    # Outer U coefficients for A blocks
    # U_outer[ block_idx, term_idx ]
    # Blocks: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    U_outer = np.array([
        [1, 0, 1, 0], # M0: A00 + A11
        [0, 0, 1, 1], # M1: A10 + A11
        [1, 0, 0, 0], # M2: A00
        [0, 0, 0, 1], # M3: A11
        [1, 1, 0, 0], # M4: A00 + A01
        [1, 0, -1, 0],# M5: A10 - A00
        [0, 1, 0, -1] # M6: A01 - A11
    ])
    
    V_outer = np.array([
        [1, 0, 0, 1], # M0: B00 + B11
        [1, 0, 0, 0], # M1: B00
        [0, 0, 1, -1],# M2: B10 - B00
        [0, 1, 0, -1],# M3: B01 - B11
        [0, 0, 0, 1], # M4: B11
        [1, 1, 0, 0], # M5: B00 + B01
        [0, 0, 1, 1]  # M6: B10 + B11
    ])
    
    # W_outer is not needed directly, W is determined by inner composition
    # Actually, W4 is composed from W_outer and W2?
    # No, the output C is formed by the outer Strassen formulas.
    # C00 = M0 + M3 - M4 - M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    # So for each outer term k, the contribution to C blocks is determined.
    # But we need to expand each Mk into 4 scalar multiplications (inner Strassen).
    
    for k in range(7):
        for l in range(7):
            col = k * 7 + l
            
            # U4 for this term
            # U_outer[k] gives weights for A blocks.
            # Each A block is 2x2.
            # U2[:, l] gives weights for elements within a block.
            # So for A block b, the weights are U_outer[k, b] * U2[:, l]
            
            for b in range(4):
                weight = U_outer[k, b]
                if weight != 0:
                    # Block b is at index (b//2, b%2) in A
                    # A block (i1, j1) corresponds to A[i1*2:i1*2+2, j1*2:j1*2+2]
                    i1, j1 = b // 2, b % 2
                    for i2 in range(2):
                        for j2 in range(2):
                            idx = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                            # U2 index for (i2, j2) is i2 * 2 + j2
                            u2_idx = i2 * 2 + j2
                            U4[idx, col] += weight * U2[u2_idx, l]
            
            # V4 for this term
            for b in range(4):
                weight = V_outer[k, b]
                if weight != 0:
                    i1, j1 = b // 2, b % 2
                    for i2 in range(2):
                        for j2 in range(2):
                            # V index is j*4+k. Here j is row of B, k is col of B.
                            # B block (i1, j1) is B[i1*2:i1*2+2, j1*2:j1*2+2]
                            # Row j = i1*2 + i2, Col k = j1*2 + j2
                            # V index = j * 4 + k = (i1*2 + i2) * 4 + (j1*2 + j2)
                            v_idx = (i1 * 2 + i2) * 4 + (j1 * 2 + j2)
                            v2_idx = i2 * 2 + j2
                            V4[v_idx, col] += weight * V2[v2_idx, l]
                            
            # W4 for this term
            # W is determined by the outer Strassen output formulas.
            # C blocks:
            # C00 (b=0) gets M0, M3, -M4, -M6
            # C01 (b=1) gets M2, M4
            # C10 (b=2) gets M1, M3
            # C11 (b=3) gets M0, -M1, M2, M5
            
            w_coeff = 0
            if k == 0:
                if l == 0: # C00
                    pass
                # This logic is wrong. W4 depends on both k and l.
                # The inner Strassen W2 gives the contribution of Mk to the elements of C_block.
                # The outer Strassen gives which C_block Mk contributes to.
                
            # Let's re-evaluate W4 construction.
            # For a fixed k, Mk contributes to certain C blocks with certain signs.
            # Let C_block_weights[b] be the coefficient of Mk in C_b.
            C_block_weights = [0, 0, 0, 0]
            if k == 0: C_block_weights = [1, 0, 0, 1] # M0 in C00, C11
            if k == 1: C_block_weights = [0, 0, 1, -1] # M1 in C10, C11
            if k == 2: C_block_weights = [0, 1, 0, 1] # M2 in C01, C11
            if k == 3: C_block_weights = [1, 0, 1, 0] # M3 in C00, C10
            if k == 4: C_block_weights = [-1, 1, 0, 0] # M4 in C00, C01
            if k == 5: C_block_weights = [0, 0, 0, 1] # M5 in C11
            if k == 6: C_block_weights = [-1, 0, 0, 0] # M6 in C00
            
            for b in range(4):
                weight = C_block_weights[b]
                if weight != 0:
                    i1, j1 = b // 2, b % 2
                    for i2 in range(2):
                        for j2 in range(2):
                            # W index is k*4+i. Here k is col of B, i is row of A.
                            # C block (i1, j1) is C[i1*2:i1*2+2, j1*2:j1*2+2]
                            # Row i = i1*2 + i2, Col k = j1*2 + j2
                            # W index = k * 4 + i = (j1*2 + j2) * 4 + (i1*2 + i2)
                            w_idx = (j1 * 2 + j2) * 4 + (i1 * 2 + i2)
                            # W2 index for (k2, i2) is k2 * 2 + i2
                            # Here k2 = j2, i2 = i2
                            w2_idx = j2 * 2 + i2
                            W4[w_idx, col] += weight * W2[w2_idx, l]
                            
    # 2. Decomposition for 4x4x4 (second block, columns 4..7 of B)
    # Same as above, but V indices are shifted by 4 in the column index of B.
    # V index = j*9 + k. For k=4..7, it's j*9 + k.
    # In V4, the index was j*4 + k (k=0..3).
    # So we need to map V4 indices to the full V indices.
    
    V4b = np.zeros((16, 49))
    for j in range(4):
        for k in range(4):
            # V4 index
            idx4 = j * 4 + k
            # Full V index
            idx_full = j * 9 + (k + 4)
            V4b[idx_full, :] = V4[idx4, :]
            
    # W4 is the same, because W index is k*n+i.
    # For the second 4x4x4 block, the output columns are 4..7.
    # W index = k*4 + i. Here k is the column index of B (and C).
    # So for the second block, k=4..7.
    # W index = (k+4)*4 + i? No.
    # The W index is global.
    # For the first block, k=0..3. W index = k*4 + i.
    # For the second block, k=4..7. W index = k*4 + i.
    # So W4b is just W4 with shifted rows?
    # No, W4 rows are already indexed by k*4+i.
    # If we use the same W4, it will write to k=0..3.
    # We need W4b to write to k=4..7.
    W4b = np.zeros((36, 49)) # p*n = 9*4 = 36
    for k in range(4):
        for i in range(4):
            idx4 = k * 4 + i
            idx_full = (k + 4) * 4 + i
            W4b[idx_full, :] = W4[idx4, :]
            
    # 3. Decomposition for 4x4x1 (columns 8 of B)
    # Rank 16.
    # U4x1 (16x16), V4x1 (4x16), W4x1 (4x16)
    U4x1 = np.zeros((16, 16))
    V4x1 = np.zeros((4, 16)) # m*p_part = 4*1 = 4
    W4x1 = np.zeros((4, 16)) # p_part*n = 1*4 = 4
    
    for i in range(4):
        for j in range(4):
            col = i * 4 + j
            U4x1[i * 4 + j, col] = 1
            V4x1[j, col] = 1
            W4x1[i, col] = 1
            
    # Assemble full U, V, W
    # U: 16 x (49+49+16) = 16 x 114
    # V: 36 x 114
    # W: 36 x 114
    
    R = 114
    U = np.zeros((16, R))
    V = np.zeros((36, R))
    W = np.zeros((36, R))
    
    # Block 1: 4x4x4 (k=0..3)
    U[:, 0:49] = U4
    V[:, 0:49] = V4
    W[:, 0:49] = W4
    
    # Block 2: 4x4x4 (k=4..7)
    U[:, 49:98] = U4
    V[:, 49:98] = V4b
    W[:, 49:98] = W4b
    
    # Block 3: 4x4x1 (k=8)
    U[:, 98:114] = U4x1
    # V4x1 is 4x16. Full V index for k=8 is j*9+8.
    for j in range(4):
        idx_full = j * 9 + 8
        V[idx_full, 98:114] = V4x1[j, :]
    # W4x1 is 4x16. Full W index for k=8 is 8*4+i = 32+i.
    for i in range(4):
        idx_full = 8 * 4 + i
        W[idx_full, 98:114] = W4x1[i, :]
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
