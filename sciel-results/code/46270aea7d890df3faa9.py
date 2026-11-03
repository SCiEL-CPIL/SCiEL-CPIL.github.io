import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Winograd F2x2x2 factors
    # U: 4x5, V: 4x5, W: 4x5
    U_w = [
        [1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 1, 1],
        [1, 1, 0, 0, 1]
    ]
    V_w = [
        [1, 1, 1, 0, 0],
        [0, 0, -1, 1, 1],
        [1, -1, -1, 0, 0],
        [0, 1, -1, 1, 1]
    ]
    W_w = [
        [1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1],
        [1, 1, 0, 0, 1]
    ]
    
    # Helper to create 2x4x4 decomposition
    def create_2x4x4():
        # 4 blocks of 2x2x2
        # A is 2x4: indices 0,1 (left), 2,3 (right)
        # B is 4x4: indices 0,1 (TL), 2,3 (TR), 4,5 (BL), 6,7 (BR) -- Wait, B is 4x4, so 4 indices.
        # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Blocks:
        # C_left = A_L B_TL + A_R B_BL
        # C_right = A_L B_TR + A_R B_BR
        
        rank_block = 5
        total_rank = 4 * rank_block
        
        U_244 = [[0]*total_rank for _ in range(n*m)] # 2*4 = 8 rows
        V_244 = [[0]*total_rank for _ in range(m*4)] # 4*4 = 16 rows
        W_244 = [[0]*total_rank for _ in range(4*n)] # 4*2 = 8 rows
        
        # Block 1: A_L (rows 0,1) * B_TL (cols 0,1 of B? No, B is 4x4. TL is rows 0,1 cols 0,1)
        # A_L indices in U: 0,1
        # B_TL indices in V: 0,1
        # C_L indices in W: 0,1
        
        # Block 2: A_R (rows 2,3) * B_BL (rows 2,3 cols 0,1 of B)
        # A_R indices in U: 2,3
        # B_BL indices in V: 2,3
        # C_L indices in W: 0,1
        
        # Block 3: A_L (rows 0,1) * B_TR (rows 0,1 cols 2,3 of B)
        # A_L indices in U: 0,1
        # B_TR indices in V: 4,5
        # C_R indices in W: 2,3
        
        # Block 4: A_R (rows 2,3) * B_BR (rows 2,3 cols 2,3 of B)
        # A_R indices in U: 2,3
        # B_BR indices in V: 6,7
        # C_R indices in W: 2,3
        
        blocks = [
            (0, 0, 0), # A_L, B_TL, C_L
            (2, 2, 0), # A_R, B_BL, C_L
            (0, 4, 2), # A_L, B_TR, C_R
            (2, 6, 2)  # A_R, B_BR, C_R
        ]
        
        for b_idx in range(4):
            a_off, b_off, c_off = blocks[b_idx]
            col_start = b_idx * rank_block
            for k in range(rank_block):
                col = col_start + k
                for i in range(4):
                    U_244[a_off + i][col] = U_w[i][k]
                    V_244[b_off + i][col] = V_w[i][k]
                    W_244[c_off + i][col] = W_w[i][k]
                    
        return U_244, V_244, W_244

    # Create decomposition for 2x4x7
    # Split B into B1 (4x4) and B2 (4x3 padded to 4x4)
    U_244, V_244, W_244 = create_2x4x4()
    
    # We need two copies of the 2x4x4 decomposition
    # One for B1, one for B2
    rank_total = 2 * (4 * 5)
    
    U_final = [[0]*rank_total for _ in range(n*m)]
    V_final = [[0]*rank_total for _ in range(m*p)]
    W_final = [[0]*rank_total for _ in range(p*n)]
    
    # First part: B1 (cols 0-3 of B)
    for i in range(n*m):
        for k in range(20):
            U_final[i][k] = U_244[i][k]
    for j in range(m*4):
        for k in range(20):
            V_final[j][k] = V_244[j][k]
    for l in range(4*n):
        for k in range(20):
            W_final[l][k] = W_244[l][k]
            
    # Second part: B2 (cols 4-6 of B, padded to 4 cols)
    # V indices for B2: 16-19 in V_final? No, V_final has m*p = 4*7 = 28 rows.
    # B1 uses V rows 0-3. B2 uses V rows 4-6.
    # In the 2x4x4 decomposition, V has 16 rows.
    # We map B2's 4 cols (3 real, 1 pad) to V_244's 16 rows.
    # B2 col 0 -> V_244 row 0? No.
    # B2 is 4x3. We treat it as 4x4.
    # V_244 rows 0-3 correspond to B2's cols 0-3.
    # But in V_final, B2's cols are at indices 4,5,6.
    # So V_final[4] corresponds to B2 col 0, which is V_244 row 0.
    # V_final[5] corresponds to B2 col 1, which is V_244 row 1.
    # V_final[6] corresponds to B2 col 2, which is V_244 row 2.
    # V_244 row 3 (pad) is ignored.
    
    for i in range(n*m):
        for k in range(20):
            U_final[i][20+k] = U_244[i][k]
            
    for j in range(3): # Only 3 real cols in B2
        for k in range(20):
            V_final[4+j][20+k] = V_244[j][k]
            
    for l in range(4*n):
        for k in range(20):
            W_final[4*n + l][20+k] = W_244[l][k]
            
    return {"U": U_final, "V": V_final, "W": W_final}
# EVOLVE-BLOCK-END
