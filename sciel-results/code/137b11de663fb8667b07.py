"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We split m=4 into two channels of size 2.
    # Channel 0: A[0:2, 0:2] * B[0:2, 0:8] -> C[0:2, 0:8]
    # Channel 1: A[0:2, 2:4] * B[2:4, 0:8] -> C[0:2, 0:8]
    
    # Each channel is 2x2 by 2x8.
    # We decompose 2x2 by 2x8 into 4 blocks of 2x2 by 2x2.
    # Each 2x2 by 2x2 uses Strassen (rank 7).
    # Total rank per channel = 4 * 7 = 28.
    # Total rank = 28 + 28 = 56.
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2x2
    # U_strassen: 7x4 (rows are products, cols are A elements)
    # A order: A00, A01, A10, A11
    U_strassen = [
        [1, 0, 1, 0], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, -1, 0],# M3: A00 - A10
        [0, 1, 0, -1],# M4: A01 - A11
        [1, 0, 0, 0], # M5: A00
        [0, 0, 0, 1], # M6: A11
        [1, 1, 0, 0]  # M7: A00 + A01
    ]
    
    # V_strassen: 7x4 (rows are products, cols are B elements)
    # B order: B00, B01, B10, B11
    V_strassen = [
        [1, 0, 0, 1], # M1: B00 + B11
        [1, 1, 0, 0], # M2: B00 + B01
        [0, 0, 1, 1], # M3: B10 + B11
        [1, 0, -1, 0],# M4: B00 - B10
        [1, 0, -1, 0],# M5: B00 - B10
        [0, 0, 1, -1],# M6: B10 - B00
        [0, 0, 0, 1]  # M7: B11
    ]
    
    # W_strassen: 7x4 (rows are products, cols are C elements)
    # C order: C00, C01, C10, C11
    W_strassen = [
        [1, 0, 0, 1],  # M1: C00 + C11
        [0, 0, 1, 0],  # M2: C10
        [0, 0, -1, 0], # M3: -C10
        [-1, 1, 0, 0], # M4: -C00 + C01
        [1, 0, 0, -1], # M5: C00 - C11
        [0, 0, 0, -1], # M6: -C11
        [-1, 1, 0, 0]  # M7: -C00 + C01
    ]
    
    # Helper to add a block
    def add_block(channel_idx, block_idx, offset):
        # channel_idx: 0 or 1
        # block_idx: 0 to 3 (which 2x2 block of the 2x8 B matrix)
        # offset: starting column index in U, V, W
        
        # A indices for this channel
        # A is 2x4. Channel 0 uses cols 0,1. Channel 1 uses cols 2,3.
        a_col_base = channel_idx * 2
        # A rows: 0, 1
        # A elements in U are at indices: i*m + j
        # i in 0..1, j in a_col_base..a_col_base+1
        
        # B indices for this channel
        # B is 4x8. Channel 0 uses rows 0,1. Channel 1 uses rows 2,3.
        b_row_base = channel_idx * 2
        # B cols: 0..7. Block block_idx uses cols 2*block_idx, 2*block_idx+1.
        b_col_base = block_idx * 2
        
        # C indices
        # C is 8x2 (p*n). 
        # C row k, col i. Index k*n + i.
        # C rows 0..7. C cols 0..1.
        # The result of this channel adds to C[0:2, 0:8]?
        # Wait, C is 2x8. 
        # The task says W[k*n+i, r].
        # So C index is k*n + i.
        # k is 0..7 (cols of original C? No, C is 2x8, so rows 0..1, cols 0..7).
        # Wait, A is 2x4, B is 4x8. C is 2x8.
        # Standard trace convention W[k*n+i, r].
        # Usually C[i, k] = sum A[i, j] B[j, k].
        # So i is row of C (0..1), k is col of C (0..7).
        # Index in W: k*n + i.
        # So for C[i, k], index is k*2 + i.
        
        for prod_idx in range(7):
            col = offset + prod_idx
            
            # U coefficients
            for a_i in range(2):
                for a_j in range(2):
                    val = U_strassen[prod_idx][a_i * 2 + a_j]
                    if val != 0:
                        # U index: i*m + j
                        # i is 0..1
                        # j is a_col_base + a_j
                        U[a_i * m + (a_col_base + a_j)][col] += val
            
            # V coefficients
            for b_i in range(2):
                for b_j in range(2):
                    val = V_strassen[prod_idx][b_i * 2 + b_j]
                    if val != 0:
                        # V index: i*p + j? No, V is (m*p, rank).
                        # V index: j*p + k?
                        # B is m x p. B[j, k].
                        # V index: j*p + k.
                        # j is b_row_base + b_i
                        # k is b_col_base + b_j
                        V[(b_row_base + b_i) * p + (b_col_base + b_j)][col] += val
            
            # W coefficients
            for c_i in range(2):
                for c_k in range(2):
                    val = W_strassen[prod_idx][c_i * 2 + c_k]
                    if val != 0:
                        # W index: k*n + i
                        # k is b_col_base + c_k
                        # i is c_i
                        W[(b_col_base + c_k) * n + c_i][col] += val

    # Channel 0
    for b in range(4):
        add_block(0, b, b * 7)
        
    # Channel 1
    for b in range(4):
        add_block(1, b, 28 + b * 7)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
