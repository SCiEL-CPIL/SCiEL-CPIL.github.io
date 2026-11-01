"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=16
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2 x 2x2
    # U_s: 4x7, V_s: 4x7, W_s: 4x7
    # Indices: A: 00, 01, 10, 11 -> 0, 1, 2, 3
    #          B: 00, 01, 10, 11 -> 0, 1, 2, 3
    #          C: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    U_s = [
        [1, 0, 1, 0, 1, -1, 0],  # m1..m7 for A00? No, rows are A indices.
        [0, 0, 0, 0, 1, 0, 1],   # A01
        [0, 1, 0, 0, 0, 1, 0],   # A10
        [1, 1, 0, 1, 0, 0, -1]   # A11
    ]
    
    V_s = [
        [1, 1, 0, -1, 0, 1, 0],  # B00
        [0, 0, 1, 0, 0, 1, 0],   # B01
        [0, 0, 0, 1, 0, 0, 1],   # B10
        [1, 0, -1, 0, 1, 0, 1]   # B11
    ]
    
    W_s = [
        [1, 0, 0, 1, -1, 0, 1],  # C00
        [0, 0, 1, 0, 1, 0, 0],   # C01
        [0, 1, 0, 1, 0, 0, 0],   # C10
        [1, -1, 1, 0, 0, 1, 0]   # C11
    ]
    
    # Verify W_s against formulas:
    # C00 = m1 + m4 - m5 + m7 -> [1, 0, 0, 1, -1, 0, 1] Correct.
    # C01 = m3 + m5 -> [0, 0, 1, 0, 1, 0, 0] Correct.
    # C10 = m2 + m4 -> [0, 1, 0, 1, 0, 0, 0] Correct.
    # C11 = m1 - m2 + m3 + m6 -> [1, -1, 1, 0, 0, 1, 0] Correct.

    # 8 blocks of 2 columns
    num_blocks = p // 2
    for l in range(num_blocks):
        start_col = l * 7
        for k in range(7):
            col_idx = start_col + k
            
            # U is same for all blocks
            for i in range(4):
                U[i][col_idx] = U_s[i][k]
                
            # V mapping
            # Local B indices:
            # 0: 00 -> j=0, k=0
            # 1: 01 -> j=0, k=1
            # 2: 10 -> j=1, k=0
            # 3: 11 -> j=1, k=1
            # Global V index: j * p + (2*l + k_local)
            
            # B00: j=0, k_loc=0 -> 0*16 + 2l
            V[0 * p + (2 * l + 0)][col_idx] = V_s[0][k]
            # B01: j=0, k_loc=1 -> 0*16 + 2l + 1
            V[0 * p + (2 * l + 1)][col_idx] = V_s[1][k]
            # B10: j=1, k_loc=0 -> 1*16 + 2l
            V[1 * p + (2 * l + 0)][col_idx] = V_s[2][k]
            # B11: j=1, k_loc=1 -> 1*16 + 2l + 1
            V[1 * p + (2 * l + 1)][col_idx] = V_s[3][k]
            
            # W mapping
            # Local C indices:
            # 0: 00 -> i=0, k=0
            # 1: 01 -> i=1, k=0
            # 2: 10 -> i=0, k=1
            # 3: 11 -> i=1, k=1
            # Global W index: k * n + i
            
            # C00: k=2l, i=0 -> (2l)*2 + 0
            W[(2 * l + 0) * n + 0][col_idx] = W_s[0][k]
            # C01: k=2l, i=1 -> (2l)*2 + 1
            W[(2 * l + 0) * n + 1][col_idx] = W_s[1][k]
            # C10: k=2l+1, i=0 -> (2l+1)*2 + 0
            W[(2 * l + 1) * n + 0][col_idx] = W_s[2][k]
            # C11: k=2l+1, i=1 -> (2l+1)*2 + 1
            W[(2 * l + 1) * n + 1][col_idx] = W_s[3][k]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
