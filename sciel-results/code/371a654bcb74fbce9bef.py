"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=8
    
    # Step 1: Define the rank-7 decomposition for 2x2x2
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # u, v, w are lists of 7 vectors of length 4
    u_2x2 = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_2x2 = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_2x2 = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Step 2: Construct 4x4x4 decomposition using 7 blocks of 2x2x2
    # Rank will be 7 * 7 = 49
    rank_4x4x4 = 49
    U_4x4 = [[0] * rank_4x4x4 for _ in range(16)] # 4x4 = 16
    V_4x4 = [[0] * rank_4x4x4 for _ in range(16)] # 4x4 = 16
    W_4x4 = [[0] * rank_4x4x4 for _ in range(16)] # 4x4 = 16
    
    # Mapping local block index (0..3) to global indices in 4x4 matrix
    # Block (bi, bj) contains elements (2*bi + r, 2*bj + c) for r,c in 0..1
    # Global index = row * 4 + col
    def get_global_indices_4x4(bi, bj):
        indices = []
        for r in range(2):
            for c in range(2):
                row = 2 * bi + r
                col = 2 * bj + c
                indices.append(row * 4 + col)
        return indices
    
    # Local indices 0,1,2,3 correspond to blocks:
    # 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
    local_to_block = {
        0: (0, 0),
        1: (0, 1),
        2: (1, 0),
        3: (1, 1)
    }
    
    for b in range(7):
        for local_idx in range(4):
            coeff_u = u_2x2[b][local_idx]
            coeff_v = v_2x2[b][local_idx]
            coeff_w = w_2x2[b][local_idx]
            
            bi, bj = local_to_block[local_idx]
            global_indices = get_global_indices_4x4(bi, bj)
            
            for gi in global_indices:
                U_4x4[gi][b*7] = coeff_u
                V_4x4[gi][b*7] = coeff_v
                W_4x4[gi][b*7] = coeff_w
                
                # Note: The rank-7 decomposition has 7 terms.
                # Here we are iterating over the 7 Strassen products (b).
                # Each Strassen product (b) is itself a 2x2x2 multiplication?
                # No, the Strassen algorithm for 4x4 uses 7 products of 2x2 matrices.
                # Each 2x2 multiplication has rank 7.
                # So for each of the 7 Strassen terms, we need 7 scalar multiplications.
                # Total 49.
                # The index in U_4x4 should be b*7 + sub_rank?
                # Yes, I used b*7 above, but I only filled one column per b.
                # I need to fill 7 columns for each b.
                
    # Re-do Step 2 correctly
    U_4x4 = [[0] * rank_4x4x4 for _ in range(16)]
    V_4x4 = [[0] * rank_4x4x4 for _ in range(16)]
    W_4x4 = [[0] * rank_4x4x4 for _ in range(16)]
    
    for b in range(7):
        # b is the Strassen term index (0..6)
        # For this term, we perform a 2x2x2 multiplication with rank 7
        for sub_r in range(7):
            col_idx = b * 7 + sub_r
            
            for local_idx in range(4):
                coeff_u = u_2x2[sub_r][local_idx]
                coeff_v = v_2x2[sub_r][local_idx]
                coeff_w = w_2x2[sub_r][local_idx]
                
                bi, bj = local_to_block[local_idx]
                global_indices = get_global_indices_4x4(bi, bj)
                
                for gi in global_indices:
                    U_4x4[gi][col_idx] = coeff_u
                    V_4x4[gi][col_idx] = coeff_v
                    W_4x4[gi][col_idx] = coeff_w

    # Step 3: Construct 4x4x8 decomposition
    # Split B into B_L (cols 0-3) and B_R (cols 4-7)
    # C = [C_L, C_R]
    # C_L = A B_L, C_R = A B_R
    # Rank is 49 + 49 = 98
    total_rank = 98
    U = [[0] * total_rank for _ in range(16)]
    V = [[0] * total_rank for _ in range(32)] # 4x8 = 32
    W = [[0] * total_rank for _ in range(32)] # 4x8 = 32
    
    # First half: C_L = A B_L
    for i in range(16):
        for r in range(49):
            U[i][r] = U_4x4[i][r]
            V[i][r] = V_4x4[i][r]
            W[i][r] = W_4x4[i][r]
            
    # Second half: C_R = A B_R
    # V indices for B_R: local col j (0..3) maps to global col j+4
    # V_4x4 index is j*4 + k? No, V_4x4 is 16x49.
    # V_4x4[row][col] where row = j*4 + k?
    # In 4x4x4, V is m x p = 4 x 4. Index = j*4 + k.
    # In 4x4x8, V is m x p = 4 x 8. Index = j*8 + k.
    # For B_R, k_global = k_local + 4.
    # So V_4x4 index (j, k_local) maps to V index (j, k_local + 4).
    # V_4x4 row index = j*4 + k_local.
    # We need to map this to V row index = j*8 + (k_local + 4).
    
    for i in range(16):
        for r in range(49):
            U[i][r + 49] = U_4x4[i][r]
            
            # Map V
            # V_4x4[i][r] corresponds to B element at (i//4, i%4) in the 4x4 block?
            # No, i is the row index of V_4x4, which is 0..15.
            # i = j*4 + k.
            j = i // 4
            k = i % 4
            # Global V index for B_R:
            v_row = j * 8 + (k + 4)
            V[v_row][r + 49] = V_4x4[i][r]
            
            # Map W
            # W_4x4 is for C_L. C_L cols 0-3.
            # W index is k*4 + i.
            # In full W, C_L is cols 0-3. So index is k*4 + i.
            # This is the same as W_4x4 index.
            W[i][r + 49] = W_4x4[i][r]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
