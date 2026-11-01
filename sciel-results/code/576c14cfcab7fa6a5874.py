"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank 14 decomposition using Strassen's algorithm for two 2x2 blocks
    n, m, p = DIMENSIONS
    rank = 14
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U rows: 00, 01, 10, 11
    # V rows for block 1 (k=0,1): 00, 01, 10, 11 -> indices 0, 1, 4, 5
    # V rows for block 2 (k=2,3): 00, 01, 10, 11 -> indices 2, 3, 6, 7
    # W rows for block 1 (k=0,1): 00, 01, 10, 11 -> indices 0, 1, 2, 3
    # W rows for block 2 (k=2,3): 00, 01, 10, 11 -> indices 4, 5, 6, 7
    
    # Strassen terms M1..M7
    # U cols
    U_cols = [
        [1, 0, 0, 1],  # M1: a00 + a11
        [0, 0, 1, 1],  # M2: a10 + a11
        [1, 0, 0, 0],  # M3: a00
        [0, 0, 0, 1],  # M4: a11
        [1, 1, 0, 0],  # M5: a00 + a01
        [-1, 0, 1, 0], # M6: a10 - a00
        [0, 1, 0, -1]  # M7: a01 - a11
    ]
    
    # V cols for one block (indices j*k)
    # b00, b01, b10, b11
    V_cols_block = [
        [1, 0, 0, 1],  # M1: b00 + b11
        [1, 0, 0, 0],  # M2: b00
        [0, 1, 0, -1], # M3: b01 - b11
        [-1, 0, 1, 0], # M4: b10 - b00
        [0, 0, 0, 1],  # M5: b11
        [1, 1, 0, 0],  # M6: b00 + b01
        [0, 0, 1, 1]   # M7: b10 + b11
    ]
    
    # W cols for one block (indices k*i)
    # c00, c01, c10, c11
    W_cols_block = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1],  # M3
        [1, 0, 1, 0],  # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 0]   # M7
    ]
    
    # Fill U
    for r in range(7):
        for i in range(2):
            for j in range(2):
                idx = i * m + j
                U[idx][r] = U_cols[r][idx]
                U[idx][r + 7] = U_cols[r][idx]
                
    # Fill V
    # Block 1: k in 0,1. Global index j*4 + k
    # Block 2: k in 2,3. Global index j*4 + k
    for r in range(7):
        for j in range(2):
            for k_local in range(2):
                # Local index in V_cols_block is j*2 + k_local
                val = V_cols_block[r][j * 2 + k_local]
                
                # Block 1
                k_global = k_local
                idx_v = j * p + k_global
                V[idx_v][r] = val
                
                # Block 2
                k_global2 = 2 + k_local
                idx_v2 = j * p + k_global2
                V[idx_v2][r + 7] = val
                
    # Fill W
    # Block 1: k in 0,1. Global index k*2 + i
    # Block 2: k in 2,3. Global index k*2 + i
    for r in range(7):
        for k_local in range(2):
            for i in range(2):
                # Local index in W_cols_block is k_local*2 + i
                val = W_cols_block[r][k_local * 2 + i]
                
                # Block 1
                k_global = k_local
                idx_w = k_global * n + i
                W[idx_w][r] = val
                
                # Block 2
                k_global2 = 2 + k_local
                idx_w2 = k_global2 * n + i
                W[idx_w2][r + 7] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
