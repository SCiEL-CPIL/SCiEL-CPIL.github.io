"""Program for matrix multiplication tensor decomposition: 2x2 by 2x16."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8 blocks of width 2
    rank = 7 * num_blocks  # 56
    
    # Initialize matrices with zeros
    # U: (n*m, rank) = (4, 56)
    # V: (m*p, rank) = (32, 56)
    # W: (p*n, rank) = (32, 56)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 Strassen factors for 2x2 matrix multiplication.
    # Each factor is a tuple (u_indices, v_indices, w_indices)
    # u_indices: list of (row_index_in_U, coeff) where row = i*m + j
    # v_indices: list of (row_index_in_V, coeff) where row = j*p + k_local
    # w_indices: list of (row_index_in_W, coeff) where row = k_local*n + i
    
    # Note: k_local is 0 or 1 relative to the block start.
    # We will add the block offset (2*t) to k_local when filling V and W.
    
    strassen_factors = [
        # s=0: M1 = (a00+a11)(b00+b11)
        {
            'u': [(0, 1), (3, 1)],  # a00 (0*2+0), a11 (1*2+1)
            'v': [(0, 1), (16, 1)], # b00 (0*16+0), b11 (1*16+1) -> relative k: 0, 1
            'w': [(0, 1), (2, 1)]   # c00 (0*2+0), c11 (1*2+1) -> relative k: 0, 1
        },
        # s=1: M2 = (a10+a11)b00
        {
            'u': [(2, 1), (3, 1)],  # a10 (1*2+0), a11 (1*2+1)
            'v': [(0, 1)],          # b00
            'w': [(2, 1), (3, -1)]  # c10 (0*2+1), c11 (1*2+1)
        },
        # s=2: M3 = a00(b01-b11)
        {
            'u': [(0, 1)],          # a00
            'v': [(1, 1), (17, -1)],# b01 (0*16+1), b11 (1*16+1)
            'w': [(1, 1), (3, 1)]   # c01 (0*2+1), c11 (1*2+1)
        },
        # s=3: M4 = a11(b10-b00)
        {
            'u': [(3, 1)],          # a11
            'v': [(16, 1), (0, -1)],# b10 (1*16+0), b00 (0*16+0)
            'w': [(0, 1), (2, 1)]   # c00 (0*2+0), c10 (0*2+1)
        },
        # s=4: M5 = (a00+a01)b11
        {
            'u': [(0, 1), (1, 1)],  # a00, a01
            'v': [(17, 1)],         # b11
            'w': [(0, -1), (1, 1)]  # c00, c01
        },
        # s=5: M6 = (a10-a00)(b00+b01)
        {
            'u': [(2, 1), (0, -1)], # a10, a00
            'v': [(0, 1), (1, 1)],  # b00, b01
            'w': [(3, 1)]           # c11
        },
        # s=6: M7 = (a01-a11)(b10+b11)
        {
            'u': [(1, 1), (3, -1)], # a01, a11
            'v': [(16, 1), (17, 1)],# b10, b11
            'w': [(0, 1)]           # c00
        }
    ]
    
    # Fill the matrices
    for t in range(num_blocks):
        block_start = 2 * t
        for s in range(7):
            col = t * 7 + s
            factor = strassen_factors[s]
            
            # Fill U
            for u_row, coeff in factor['u']:
                U[u_row][col] = coeff
                
            # Fill V
            # V row index = j * p + k_global
            # k_global = block_start + k_local
            # The 'v' list contains (k_local_offset_from_j_start, coeff) ? 
            # No, in my definition above for strassen_factors, I used absolute indices for a single block starting at 0.
            # Let's re-verify the indices in the list above.
            # For s=0, 'v': [(0, 1), (16, 1)].
            # If block_start=0, rows are 0 and 16.
            # If block_start=2, rows should be 2 and 18.
            # So we need to add block_start to the row index in V?
            # V row = j * p + k.
            # In the list, 0 corresponds to j=0, k=0. 16 corresponds to j=1, k=0.
            # So the stored value is the row index for block 0.
            # For block t, the row index becomes stored_row + block_start.
            
            for v_row_base, coeff in factor['v']:
                v_row = v_row_base + block_start
                V[v_row][col] = coeff
                
            # Fill W
            # W row index = k_global * n + i
            # k_global = block_start + k_local
            # In the list, 'w' indices are for block 0.
            # e.g., s=0, 'w': [(0, 1), (2, 1)].
            # 0 = k=0, i=0. 2 = k=1, i=0? No.
            # W row = k * n + i.
            # For k=0, i=0 -> 0.
            # For k=1, i=1 -> 1*2 + 1 = 3?
            # Let's check s=0 'w': [(0, 1), (2, 1)].
            # If it's c00 and c11.
            # c00: k=0, i=0 -> row 0.
            # c11: k=1, i=1 -> row 1*2 + 1 = 3.
            # But the list has 2. 
            # Wait, in the seed code:
            # W[(2 * t + k_local) * n + i]
            # For M1: W entries were (0,0,1) and (1,1,1).
            # k_local=0, i=0 -> row 0.
            # k_local=1, i=1 -> row 3.
            # My list has (0,1) and (2,1). This is wrong.
            
    # Let's correct the Strassen factor definitions to use (k_local, i, coeff) for W
    # and (j, k_local, coeff) for V
    # and (i, j, coeff) for U
    
    strassen_factors_v2 = [
        # s=0: M1
        {
            'u': [(0, 0, 1), (1, 1, 1)], # (i,j,c): a00, a11
            'v': [(0, 0, 1), (1, 1, 1)], # (j,k,c): b00, b11
            'w': [(0, 0, 1), (1, 1, 1)]  # (k,i,c): c00, c11
        },
        # s=1: M2
        {
            'u': [(1, 0, 1), (1, 1, 1)], # a10, a11
            'v': [(0, 0, 1)],            # b00
            'w': [(0, 1, 1), (1, 1, -1)] # c10 (k=0,i=1), c11 (k=1,i=1)
        },
        # s=2: M3
        {
            'u': [(0, 0, 1)],            # a00
            'v': [(0, 1, 1), (1, 1, -1)],# b01, b11
            'w': [(1, 0, 1), (1, 1, 1)]  # c01 (k=1,i=0), c11 (k=1,i=1)
        },
        # s=3: M4
        {
            'u': [(1, 1, 1)],            # a11
            'v': [(1, 0, 1), (0, 0, -1)],# b10, b00
            'w': [(0, 0, 1), (0, 1, 1)]  # c00 (k=0,i=0), c10 (k=0,i=1)
        },
        # s=4: M5
        {
            'u': [(0, 0, 1), (0, 1, 1)], # a00, a01
            'v': [(1, 1, 1)],            # b11
            'w': [(0, 0, -1), (1, 0, 1)] # c00 (k=0,i=0), c01 (k=1,i=0)
        },
        # s=5: M6
        {
            'u': [(1, 0, 1), (0, 0, -1)],# a10, a00
            'v': [(0, 0, 1), (0, 1, 1)], # b00, b01
            'w': [(1, 1, 1)]             # c11 (k=1,i=1)
        },
        # s=6: M7
        {
            'u': [(0, 1, 1), (1, 1, -1)],# a01, a11
            'v': [(1, 0, 1), (1, 1, 1)], # b10, b11
            'w': [(0, 0, 1)]             # c00 (k=0,i=0)
        }
    ]
    
    # Reset matrices to be safe
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t in range(num_blocks):
        block_start = 2 * t
        for s in range(7):
            col = t * 7 + s
            factor = strassen_factors_v2[s]
            
            # Fill U: row = i*m + j
            for i, j, coeff in factor['u']:
                U[i * m + j][col] = coeff
                
            # Fill V: row = j*p + (block_start + k_local)
            for j, k_local, coeff in factor['v']:
                V[j * p + block_start + k_local][col] = coeff
                
            # Fill W: row = (block_start + k_local)*n + i
            for k_local, i, coeff in factor['w']:
                W[(block_start + k_local) * n + i][col] = coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
