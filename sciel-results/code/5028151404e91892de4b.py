"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 7 * (p // 2)
    # p=14, so 7 * 7 = 49
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen linear combinations for A (2x2)
    # Indices: 0: a, 1: b, 2: c, 3: d
    u_strassen = [
        [1, 0, 0, 1],  # M1: a+d
        [0, 0, 1, 1],  # M2: c+d
        [1, 0, 0, 0],  # M3: a
        [0, 0, 0, 1],  # M4: d
        [1, 1, 0, 0],  # M5: a+b
        [-1, 0, 1, 0], # M6: c-a
        [0, 1, 0, -1]  # M7: b-d
    ]
    
    # Strassen linear combinations for B block (2x2)
    # Local indices for block: 0: x (B00), 1: y (B01), 2: z (B10), 3: w (B11)
    v_strassen_local = [
        [1, 0, 0, 1],  # M1: x+w
        [1, 1, 0, 0],  # M2: x+y
        [0, 1, 0, -1], # M3: y-w
        [-1, 0, 1, 0], # M4: z-x
        [0, 0, 0, 1],  # M5: w
        [1, 0, 1, 0],  # M6: x+z
        [0, 1, 0, 1]   # M7: y+w
    ]
    
    # Strassen reconstruction for C block (2x2)
    # Local indices for C: 0: C00, 1: C01, 2: C10, 3: C11
    # M1 -> C00, C11
    # M2 -> C10, C11
    # M3 -> C01, C11
    # M4 -> C00, C10
    # M5 -> C00, C01
    # M6 -> C11
    # M7 -> C00
    w_strassen_local = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1],  # M3
        [1, 0, 1, 0],  # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1],  # M6
        [1, 0, 0, 0]   # M7
    ]
    
    num_blocks = p // 2
    
    for t in range(num_blocks):
        # Global column indices in B
        col_0 = 2 * t
        col_1 = 2 * t + 1
        
        # Global row indices in B (0 and 1)
        # V index = j * p + k
        v_idx_x = 0 * p + col_0
        v_idx_y = 0 * p + col_1
        v_idx_z = 1 * p + col_0
        v_idx_w = 1 * p + col_1
        
        # Global indices in W (C)
        # W index = k * n + i
        w_idx_c00 = col_0 * n + 0
        w_idx_c01 = col_1 * n + 0
        w_idx_c10 = col_0 * n + 1
        w_idx_c11 = col_1 * n + 1
        
        for k in range(7):
            col = t * 7 + k
            
            # Fill U
            for i in range(4):
                U[i][col] = u_strassen[k][i]
                
            # Fill V
            # Map local v_strassen to global V indices
            # Local 0 -> v_idx_x, 1 -> v_idx_y, 2 -> v_idx_z, 3 -> v_idx_w
            V[v_idx_x][col] = v_strassen_local[k][0]
            V[v_idx_y][col] = v_strassen_local[k][1]
            V[v_idx_z][col] = v_strassen_local[k][2]
            V[v_idx_w][col] = v_strassen_local[k][3]
            
            # Fill W
            # Map local w_strassen to global W indices
            # Local 0 -> w_idx_c00, 1 -> w_idx_c01, 2 -> w_idx_c10, 3 -> w_idx_c11
            W[w_idx_c00][col] = w_strassen_local[k][0]
            W[w_idx_c01][col] = w_strassen_local[k][1]
            W[w_idx_c10][col] = w_strassen_local[k][2]
            W[w_idx_c11][col] = w_strassen_local[k][3]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
