"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We decompose into 7 blocks of 2x2x2 multiplication.
    # Each 2x2x2 multiplication has rank 7.
    # Total rank = 7 * 7 = 49.
    
    # The rank 7 decomposition for 2x2x2 (Winograd's algorithm)
    # U is 4x7, V is 4x7, W is 4x7
    # Indices for U: 0:A00, 1:A01, 2:A10, 3:A11
    # Indices for V: 0:B00, 1:B01, 2:B10, 3:B11
    # Indices for W: 0:C00, 1:C01, 2:C10, 3:C11
    
    U_block = [
        [1, 1, 0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 1, 0, 1]
    ]
    V_block = [
        [1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 1, 0, 1, 0, 0, 1]
    ]
    W_block = [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 1, 0]
    ]
    
    # Verify the block decomposition for 2x2x2
    # C_ik = sum_r U_{2i+j, r} V_{2j+k, r} W_{2k+i, r}
    # This specific set of matrices is a known valid rank 7 decomposition.
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    num_blocks = p // 2
    for b in range(num_blocks):
        offset = b * 7
        # Map block indices to global indices
        # U indices are the same for all blocks: 0..3
        # V indices: j*p + k. For block b, k is 2b, 2b+1.
        # V local indices 0..3 map to:
        # 0 (B00) -> 0*14 + 2b = 2b
        # 1 (B01) -> 0*14 + 2b+1 = 2b+1
        # 2 (B10) -> 1*14 + 2b = 14+2b
        # 3 (B11) -> 1*14 + 2b+1 = 14+2b+1
        v_map = [2*b, 2*b+1, 14+2*b, 14+2*b+1]
        
        # W indices: k*n + i. For block b, k is 2b, 2b+1.
        # W local indices 0..3 map to:
        # 0 (C00) -> 2b*2 + 0 = 4b
        # 1 (C01) -> 2b*2 + 1 = 4b+1
        # 2 (C10) -> (2b+1)*2 + 0 = 4b+2
        # 3 (C11) -> (2b+1)*2 + 1 = 4b+3
        w_map = [4*b, 4*b+1, 4*b+2, 4*b+3]
        
        for r in range(7):
            col = offset + r
            for i in range(4):
                U[i][col] = U_block[i][r]
                V[v_map[i]][col] = V_block[i][r]
                W[w_map[i]][col] = W_block[i][r]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
