"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a decomposition based on Strassen's algorithm for the 4x4x4 part.
    # The rank for 4x4x4 is 28 (7 Strassen products * 4 schoolbook multiplications per product).
    # For 4x4x8, we split B into two 4x4 blocks, so rank is 2 * 28 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)] # 16 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 32 x 56
    
    # Helper to get block indices
    def get_block_indices(block):
        # block is 0, 1, 2, 3 corresponding to 11, 12, 21, 22
        # 11: i in 0,1 j in 0,1
        # 12: i in 0,1 j in 2,3
        # 21: i in 2,3 j in 0,1
        # 22: i in 2,3 j in 2,3
        if block == 0: return [(0,0), (0,1), (1,0), (1,1)]
        if block == 1: return [(0,2), (0,3), (1,2), (1,3)]
        if block == 2: return [(2,0), (2,1), (3,0), (3,1)]
        if block == 3: return [(2,2), (2,3), (3,2), (3,3)]

    # Define the 7 Strassen products
    # Each product is defined by:
    # A_blocks: list of (block_id, coefficient)
    # B_blocks: list of (block_id, coefficient)
    # C_blocks: list of (block_id, coefficient)
    
    products = [
        # M1
        {
            'A': [(0, 1), (3, 1)],
            'B': [(0, 1), (3, 1)],
            'C': [(0, 1), (3, 1)] # C11 and C22
        },
        # M2
        {
            'A': [(2, 1), (3, 1)],
            'B': [(0, 1)],
            'C': [(2, 1)] # C21
        },
        # M3
        {
            'A': [(0, 1)],
            'B': [(1, 1), (3, -1)],
            'C': [(1, 1), (3, 1)] # C12 and C22
        },
        # M4
        {
            'A': [(3, 1)],
            'B': [(2, 1), (0, -1)],
            'C': [(0, 1), (2, 1)] # C11 and C21
        },
        # M5
        {
            'A': [(0, 1), (1, 1)],
            'B': [(3, 1)],
            'C': [(0, -1), (1, 1)] # C11 (-1) and C12 (1)
        },
        # M6
        {
            'A': [(2, 1), (0, -1)],
            'B': [(0, 1), (1, 1)],
            'C': [(3, 1)] # C22
        },
        # M7
        {
            'A': [(1, 1), (3, -1)],
            'B': [(2, 1), (3, 1)],
            'C': [(0, 1), (1, 1)] # C11 and C12
        }
    ]
    
    col = 0
    for prod in products:
        # For each of the 4 "schoolbook" multiplications within this Strassen product
        # We iterate j from 0 to 3 (the inner dimension of the 4x4 matrix product)
        for j in range(4):
            # Construct U vector for this rank-1 term
            # U corresponds to the linear form in A for the i,j entry of the A-factor matrix
            # The A-factor matrix is formed by the sum of blocks.
            # Entry (i,j) of the A-factor matrix is non-zero only if (i,j) is in one of the A_blocks.
            # Wait, the Strassen product is M = L_A L_B.
            # L_A is a 4x4 matrix. L_B is a 4x4 matrix.
            # M[i,k] = sum_j L_A[i,j] L_B[j,k].
            # So for a fixed j, we have a term L_A[i,j] * L_B[j,k].
            # L_A[i,j] is a linear form in A.
            # L_B[j,k] is a linear form in B.
            
            # Determine L_A[i,j] for all i
            # L_A is the sum of coefficients * Block.
            # A block is a 2x2 submatrix.
            # If (i,j) is in a block, the coefficient is the block's coefficient.
            
            u_vec = [0] * (n * m) # 16
            for a_block_id, a_coeff in prod['A']:
                # Check if (i,j) is in this block for any i?
                # We need to fill u_vec for all i.
                # u_vec[i*m + j] should be the coefficient of A[i,j] in L_A[i,j].
                # L_A[i,j] is non-zero only if (i,j) is in one of the A_blocks.
                # If (i,j) is in block a_block_id, then L_A[i,j] = a_coeff * A[i,j].
                # So u_vec[i*m + j] = a_coeff.
                
                # Get indices in this block
                # We need to map (i,j) to block_id.
                # Block 0: i in 0,1 j in 0,1
                # Block 1: i in 0,1 j in 2,3
                # Block 2: i in 2,3 j in 0,1
                # Block 3: i in 2,3 j in 2,3
                
                # For a fixed j, which blocks contain this j?
                # If j in 0,1: blocks 0, 2
                # If j in 2,3: blocks 1, 3
                
                # We iterate i to fill u_vec
                for i in range(4):
                    # Determine which block (i,j) belongs to
                    if i < 2:
                        if j < 2:
                            block_idx = 0
                        else:
                            block_idx = 1
                    else:
                        if j < 2:
                            block_idx = 2
                        else:
                            block_idx = 3
                    
                    # Check if this block is in prod['A']
                    for a_bid, a_c in prod['A']:
                        if a_bid == block_idx:
                            u_vec[i * m + j] = a_c
                            break # Only one block per (i,j)
            
            # Construct V vector for this rank-1 term
            # v_vec[j*p + k] should be the coefficient of B[j,k] in L_B[j,k].
            v_vec = [0] * (m * p) # 32
            # L_B[j,k] depends on which block (j,k) is in.
            # For a fixed j, we iterate k.
            for k in range(8):
                # Determine which block (j,k) belongs to in the 4x4 B matrix?
                # Wait, B is 4x8.
                # The Strassen algorithm is for 4x4 B.
                # We are applying it to the first 4x4 block of B (cols 0-3) and the second (cols 4-7).
                # So for the first 28 terms (cols 0-3), k is 0-3.
                # For the next 28 terms (cols 4-7), k is 4-7.
                
                # This loop is for the current Strassen product 'prod'.
                # We are generating 4 rank-1 terms for this product.
                # These terms correspond to the multiplication of the 4x4 A-factor and 4x4 B-factor.
                # The B-factor is 4x4.
                # So k in L_B is 0-3.
                
                # We are filling v_vec for the current column 'col'.
                # The current column corresponds to one of the 4 inner products j.
                
                # For the first 28 columns (block 1 of B):
                # v_vec[j*8 + k] for k in 0-3.
                
                # For the next 28 columns (block 2 of B):
                # v_vec[j*8 + k] for k in 4-7.
                
                # We will handle the two blocks separately in the outer loop.
                # So here, we just fill the 4x4 part.
                
                # Determine block for (j, k_local) where k_local is 0-3
                k_local = k % 4 # This is wrong, k is 0-3 for the 4x4 block.
                
                # Let's restructure.
                # I will generate the 28 terms for the first B block, then the 28 for the second.
                
                pass
            
            # This approach is getting messy. Let's simplify.
            # I will construct the 4x4x4 factors (U4, V4, W4) of size 16x28, 16x28, 16x28.
            # Then I will map them to the 4x4x8 factors.
            
    # Let's restart the construction logic inside the function to be cleaner.
    
    def construct_4x4x4():
        rank4 = 28
        U4 = [[0] * rank4 for _ in range(16)]
        V4 = [[0] * rank4 for _ in range(16)]
        W4 = [[0] * rank4 for _ in range(16)]
        
        col4 = 0
        for prod in products:
            for j in range(4):
                # U4 vector
                u_vec = [0] * 16
                for i in range(4):
                    # Determine block of (i,j)
                    if i < 2:
                        b_idx = 0 if j < 2 else 1
                    else:
                        b_idx = 2 if j < 2 else 3
                    
                    for a_bid, a_c in prod['A']:
                        if a_bid == b_idx:
                            u_vec[i * 4 + j] = a_c
                            break
                
                # V4 vector
                v_vec = [0] * 16
                for k in range(4):
                    # Determine block of (j,k)
                    if j < 2:
                        b_idx = 0 if k < 2 else 1
                    else:
                        b_idx = 2 if k < 2 else 3
                    
                    for b_bid, b_c in prod['B']:
                        if b_bid == b_idx:
                            v_vec[j * 4 + k] = b_c
                            break
                
                # W4 vector
                w_vec = [0] * 16
                for i in range(4):
                    for k in range(4):
                        # Determine block of (k,i) in C?
                        # C block indices:
                        # C11: i in 0,1 k in 0,1 -> block 0
                        # C12: i in 0,1 k in 2,3 -> block 1
                        # C21: i in 2,3 k in 0,1 -> block 2
                        # C22: i in 2,3 k in 2,3 -> block 3
                        
                        # W index is k * 4 + i
                        if i < 2:
                            c_idx = 0 if k < 2 else 1
                        else:
                            c_idx = 2 if k < 2 else 3
                        
                        for c_bid, c_c in prod['C']:
                            if c_bid == c_idx:
                                w_vec[k * 4 + i] = c_c
                                break
                
                # Fill U4, V4, W4
                for r in range(16):
                    U4[r][col4] = u_vec[r]
                    V4[r][col4] = v_vec[r]
                    W4[r][col4] = w_vec[r]
                
                col4 += 1
        
        return U4, V4, W4

    U4, V4, W4 = construct_4x4x4()
    
    # Map to 4x4x8
    # U is 16x56
    # V is 32x56
    # W is 32x56
    
    # First 28 columns: B block 1 (cols 0-3)
    for c in range(28):
        for r in range(16):
            U[r][c] = U4[r][c]
            # V index: j * 8 + k. k is 0-3.
            # V4 index: j * 4 + k.
            # So V[r][c] = V4[r][c] is not correct because row indices differ.
            # V4 row index is j * 4 + k.
            # V row index is j * 8 + k.
            # We need to map V4 rows to V rows.
            
            # Let's do it row by row.
            
    # Better mapping:
    for c in range(28):
        for r in range(16):
            U[r][c] = U4[r][c]
            
            # V mapping
            # r in V4 is j * 4 + k
            # r in V is j * 8 + k
            # So for each r in 0..15:
            # j = r // 4
            # k = r % 4
            # V_row = j * 8 + k
            # V[V_row][c] = V4[r][c]
            
            j = r // 4
            k = r % 4
            V_row = j * 8 + k
            V[V_row][c] = V4[r][c]
            
            # W mapping
            # r in W4 is k * 4 + i
            # r in W is k * 4 + i (since n=4, p=8, W index is k * n + i = k * 4 + i)
            # Wait, W shape is p * n = 32.
            # W index is k * 4 + i.
            # In the first block, k is 0-3.
            # So W[r][c] = W4[r][c] directly.
            W[r][c] = W4[r][c]
            
    # Next 28 columns: B block 2 (cols 4-7)
    for c in range(28):
        for r in range(16):
            U[r][c + 28] = U4[r][c]
            
            # V mapping
            j = r // 4
            k = r % 4
            # k in B is 4 + k
            V_row = j * 8 + (4 + k)
            V[V_row][c + 28] = V4[r][c]
            
            # W mapping
            # k in C is 4 + k
            # W index is k * 4 + i
            # W4 index is k * 4 + i
            # So W_row = (4 + k) * 4 + i
            # r = k * 4 + i
            # k = r // 4
            # i = r % 4
            # W_row = (4 + k) * 4 + i = 16 + k * 4 + i = 16 + r
            W[16 + r][c + 28] = W4[r][c]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
