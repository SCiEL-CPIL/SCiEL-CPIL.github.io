"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 multiplication
    # u_k, v_k for k=1..7
    # Indices in 2x2 block: 00, 01, 10, 11
    # u1 = 00+11, v1 = 00+11
    # u2 = 10+11, v2 = 00
    # u3 = 00,     v3 = 01-11
    # u4 = 11,     v4 = 10-00
    # u5 = 00+01,  v5 = 11
    # u6 = 10-00,  v6 = 00+10
    # u7 = 01-11,  v7 = 10+11
    
    # w_coeffs for z_00, z_01, z_10, z_11
    # z00 = M1 + M4 - M5 + M7
    # z01 = M3 + M5
    # z10 = M2 + M4
    # z11 = M1 - M2 + M3 + M6
    w_coeffs = [
        [1, 0, 0, 1, -1, 0, 1], # z00
        [0, 0, 1, 0, 1, 0, 0],  # z01
        [0, 1, 0, 1, 0, 0, 0],  # z10
        [1, -1, 1, 0, 0, 1, 0]  # z11
    ]
    
    # Define block mappings
    # Block 1: A_L (cols 0,1) * B_11 (rows 0,1 cols 0,1) -> C_L (cols 0,1)
    # Block 2: A_R (cols 2,3) * B_21 (rows 2,3 cols 0,1) -> C_L (cols 0,1)
    # Block 3: A_L (cols 0,1) * B_12 (rows 0,1 cols 2,3) -> C_R (cols 2,3)
    # Block 4: A_R (cols 2,3) * B_22 (rows 2,3 cols 2,3) -> C_R (cols 2,3)
    
    # A indices: i in 0..1, j in 0..3
    # A_L: j=0,1. A_R: j=2,3.
    # B indices: j in 0..3, k in 0..3
    # B_11: j=0,1, k=0,1. B_21: j=2,3, k=0,1.
    # B_12: j=0,1, k=2,3. B_22: j=2,3, k=2,3.
    # C indices: i in 0..1, k in 0..3
    # C_L: k=0,1. C_R: k=2,3.
    
    blocks = [
        # (U_row_map, V_row_map, W_row_map, col_offset)
        # U_row_map: {local_idx: global_row_idx} for A
        # V_row_map: {local_idx: global_row_idx} for B
        # W_row_map: {local_idx: global_row_idx} for C
        # local indices 0:00, 1:01, 2:10, 3:11
        
        # Block 1: A_L * B_11
        ({0: 0, 1: 1, 2: 4, 3: 5}, 
         {0: 0, 1: 1, 2: 4, 3: 5}, 
         {0: 0, 1: 1, 2: 4, 3: 5}, 
         0),
         
        # Block 2: A_R * B_21
        ({0: 2, 1: 3, 2: 6, 3: 7}, 
         {0: 8, 1: 9, 2: 12, 3: 13}, 
         {0: 0, 1: 1, 2: 4, 3: 5}, 
         7),
         
        # Block 3: A_L * B_12
        ({0: 0, 1: 1, 2: 4, 3: 5}, 
         {0: 2, 1: 3, 2: 6, 3: 7}, 
         {0: 2, 1: 3, 2: 6, 3: 7}, 
         14),
         
        # Block 4: A_R * B_22
        ({0: 2, 1: 3, 2: 6, 3: 7}, 
         {0: 10, 1: 11, 2: 14, 3: 15}, 
         {0: 2, 1: 3, 2: 6, 3: 7}, 
         21)
    ]
    
    for u_map, v_map, w_map, offset in blocks:
        for k in range(7):
            col = offset + k
            # U
            if k == 0:
                U[u_map[0]][col] = 1
                U[u_map[3]][col] = 1
            elif k == 1:
                U[u_map[2]][col] = 1
                U[u_map[3]][col] = 1
            elif k == 2:
                U[u_map[0]][col] = 1
            elif k == 3:
                U[u_map[3]][col] = 1
            elif k == 4:
                U[u_map[0]][col] = 1
                U[u_map[1]][col] = 1
            elif k == 5:
                U[u_map[2]][col] = 1
                U[u_map[0]][col] = -1
            elif k == 6:
                U[u_map[1]][col] = 1
                U[u_map[3]][col] = -1
            
            # V
            if k == 0:
                V[v_map[0]][col] = 1
                V[v_map[3]][col] = 1
            elif k == 1:
                V[v_map[0]][col] = 1
            elif k == 2:
                V[v_map[1]][col] = 1
                V[v_map[3]][col] = -1
            elif k == 3:
                V[v_map[2]][col] = 1
                V[v_map[0]][col] = -1
            elif k == 4:
                V[v_map[3]][col] = 1
            elif k == 5:
                V[v_map[0]][col] = 1
                V[v_map[2]][col] = 1
            elif k == 6:
                V[v_map[2]][col] = 1
                V[v_map[3]][col] = 1
            
            # W
            # w_coeffs[0] for z00, [1] for z01, [2] for z10, [3] for z11
            # z00 -> W row 0 (C_00)
            # z01 -> W row 1 (C_01)
            # z10 -> W row 2 (C_10)
            # z11 -> W row 3 (C_11)
            # But w_map maps local C indices to global C rows.
            # local 0: 00, 1: 01, 2: 10, 3: 11
            
            for z_idx in range(4):
                coeff = w_coeffs[z_idx][k]
                if coeff != 0:
                    W[w_map[z_idx]][col] = coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
