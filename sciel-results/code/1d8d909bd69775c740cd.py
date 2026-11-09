"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 145.0  # 5s margin
    
    n, m, p = DIMENSIONS
    
    # Rank calculation: 3 blocks * 4 Strassen instances * 7 rank per instance = 84
    rank = 84
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Local Strassen decomposition for 2x2 x 2x2 = 2x2
    # U, V, W are 4x7 matrices
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1: (X00+X11)(Y00+Y11) -> Z00, Z11
    # M2: (X10+X11)Y00 -> Z10
    # M3: X00(Y01-Y11) -> Z01, Z11
    # M4: X11(Y10-Y00) -> Z00, Z10
    # M5: (X00+X01)Y11 -> Z00, Z01, Z11
    # M6: (X10-X00)(Y00+Y01) -> Z11
    # M7: (X01-X11)(Y10+Y11) -> Z00, Z11
    
    strassen_U = [
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, -1],
        [1, 1, 0, 1, 0, 0, 0]
    ]
    strassen_V = [
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, -1, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    strassen_W = [
        [1, 0, 0, -1, 1, 0, -1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, -1, 0, -1, 1, -1]
    ]
    
    # Helper to add local matrices to global
    def add_block(U_local, V_local, W_local, 
                  u_indices, v_indices, w_indices, col_offset):
        for r in range(7):
            global_col = col_offset + r
            for i in range(4):
                # U
                if U_local[i][r] != 0:
                    U[u_indices[i]][global_col] += U_local[i][r]
                # V
                if V_local[i][r] != 0:
                    V[v_indices[i]][global_col] += V_local[i][r]
                # W
                if W_local[i][r] != 0:
                    W[w_indices[i]][global_col] += W_local[i][r]

    col_offset = 0
    
    # Loop over 3 blocks of 2 columns
    for b in range(3):
        # B columns: 2b, 2b+1
        # C columns: 2b, 2b+1
        
        # V indices for B_top (rows 0,1) and B_bot (rows 2,3)
        # V index = row * p + col
        v_top_indices = [
            0 * p + 2*b,
            0 * p + 2*b + 1,
            1 * p + 2*b,
            1 * p + 2*b + 1
        ]
        v_bot_indices = [
            2 * p + 2*b,
            2 * p + 2*b + 1,
            3 * p + 2*b,
            3 * p + 2*b + 1
        ]
        
        # W indices for C_top (rows 0,1) and C_bot (rows 2,3)
        # W index = col * n + row
        w_top_indices = [
            (2*b) * n + 0,
            (2*b + 1) * n + 0,
            (2*b) * n + 1,
            (2*b + 1) * n + 1
        ]
        w_bot_indices = [
            (2*b) * n + 2,
            (2*b + 1) * n + 2,
            (2*b) * n + 3,
            (2*b + 1) * n + 3
        ]
        
        # U indices for A_11 (rows 0,1, cols 0,1)
        u_11_indices = [
            0 * m + 0,
            0 * m + 1,
            1 * m + 0,
            1 * m + 1
        ]
        # U indices for A_12 (rows 0,1, cols 2,3)
        u_12_indices = [
            0 * m + 2,
            0 * m + 3,
            1 * m + 2,
            1 * m + 3
        ]
        # U indices for A_21 (rows 2,3, cols 0,1)
        u_21_indices = [
            2 * m + 0,
            2 * m + 1,
            3 * m + 0,
            3 * m + 1
        ]
        # U indices for A_22 (rows 2,3, cols 2,3)
        u_22_indices = [
            2 * m + 2,
            2 * m + 3,
            3 * m + 2,
            3 * m + 3
        ]
        
        # 1. A_11 * B_top -> C_top
        add_block(strassen_U, strassen_V, strassen_W,
                  u_11_indices, v_top_indices, w_top_indices, col_offset)
        col_offset += 7
        
        # 2. A_12 * B_bot -> C_top
        add_block(strassen_U, strassen_V, strassen_W,
                  u_12_indices, v_bot_indices, w_top_indices, col_offset)
        col_offset += 7
        
        # 3. A_21 * B_top -> C_bot
        add_block(strassen_U, strassen_V, strassen_W,
                  u_21_indices, v_top_indices, w_bot_indices, col_offset)
        col_offset += 7
        
        # 4. A_22 * B_bot -> C_bot
        add_block(strassen_U, strassen_V, strassen_W,
                  u_22_indices, v_bot_indices, w_bot_indices, col_offset)
        col_offset += 7
        
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
