"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: n*m x rank
    # V: m*p x rank
    # W: p*n x rank
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition vectors for 2x2x2
    # Local indices: 0=x00, 1=x01, 2=x10, 3=x11
    # Same for y, z
    # Order of w: z00, z01, z10, z11
    
    strassen_u = [
        [0, 0, 0, 1],  # M1: x11
        [0, 0, 1, 1],  # M2: x10+x11
        [1, 0, 0, 0],  # M3: x00
        [1, 0, 0, 0],  # M4: x00
        [1, 0, 0, 1],  # M5: x00+x11
        [-1, 1, 0, 0], # M6: x01-x00
        [0, 0, 1, -1]  # M7: x10-x11
    ]
    
    strassen_v = [
        [1, 0, -1, 0], # M1: y00-y10
        [0, 0, 1, 0],  # M2: y10
        [0, -1, 0, 1], # M3: y11-y01
        [1, 0, 0, 0],  # M4: y00
        [0, 1, 0, 1],  # M5: y01+y11
        [1, 0, 1, 0],  # M6: y00+y10
        [0, 1, 0, 1]   # M7: y01+y11
    ]
    
    strassen_w = [
        [-1, 0, 1, 0], # M1: -z00 + z10
        [0, 0, 1, -1], # M2: z10 - z11
        [0, 0, 0, 1],  # M3: z11
        [1, 0, 0, 0],  # M4: z00
        [1, 0, 0, 1],  # M5: z00 + z11
        [0, 1, 0, -1], # M6: z01 - z11
        [-1, 1, 0, 0]  # M7: -z00 + z01
    ]
    
    def add_strassen_instance(col_offset, x_map, y_map, z_map):
        # x_map: [global_idx_x00, global_idx_x01, global_idx_x10, global_idx_x11]
        # y_map: [global_idx_y00, global_idx_y01, global_idx_y10, global_idx_y11]
        # z_map: [global_idx_z00, global_idx_z01, global_idx_z10, global_idx_z11]
        
        for r in range(7):
            col = col_offset + r
            u_vec = strassen_u[r]
            v_vec = strassen_v[r]
            w_vec = strassen_w[r]
            
            for i in range(4):
                if u_vec[i] != 0:
                    U[x_map[i]][col] += u_vec[i]
                if v_vec[i] != 0:
                    V[y_map[i]][col] += v_vec[i]
                if w_vec[i] != 0:
                    W[z_map[i]][col] += w_vec[i]
    
    # Instance 1a: A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    # A indices: i in 0,1; j in 0,1. U_idx = i*4 + j
    # B indices: j in 0,1; k in 0,1. V_idx = j*4 + k
    # C indices: i in 0,1; k in 0,1. W_idx = k*2 + i
    
    x_map_1a = [0*4+0, 0*4+1, 1*4+0, 1*4+1] # A00, A01, A10, A11
    y_map_1a = [0*4+0, 0*4+1, 1*4+0, 1*4+1] # B00, B01, B10, B11
    z_map_1a = [0*2+0, 0*2+1, 1*2+0, 1*2+1] # C00, C01, C10, C11
    
    add_strassen_instance(0, x_map_1a, y_map_1a, z_map_1a)
    
    # Instance 1b: A[0:2, 0:2] * B[0:2, 2:4] -> C[0:2, 2:4]
    # A indices: same as 1a
    # B indices: j in 0,1; k in 2,3. V_idx = j*4 + k
    # C indices: i in 0,1; k in 2,3. W_idx = k*2 + i
    
    x_map_1b = [0*4+0, 0*4+1, 1*4+0, 1*4+1]
    y_map_1b = [0*4+2, 0*4+3, 1*4+2, 1*4+3] # B02, B03, B12, B13
    z_map_1b = [2*2+0, 2*2+1, 3*2+0, 3*2+1] # C02, C03, C12, C13
    
    add_strassen_instance(7, x_map_1b, y_map_1b, z_map_1b)
    
    # Instance 2a: A[0:2, 2:4] * B[2:4, 0:2] -> C[0:2, 0:2]
    # A indices: i in 0,1; j in 2,3. U_idx = i*4 + j
    # B indices: j in 2,3; k in 0,1. V_idx = j*4 + k
    # C indices: i in 0,1; k in 0,1. W_idx = k*2 + i
    
    x_map_2a = [0*4+2, 0*4+3, 1*4+2, 1*4+3] # A02, A03, A12, A13
    y_map_2a = [2*4+0, 2*4+1, 3*4+0, 3*4+1] # B20, B21, B30, B31
    z_map_2a = [0*2+0, 0*2+1, 1*2+0, 1*2+1]
    
    add_strassen_instance(14, x_map_2a, y_map_2a, z_map_2a)
    
    # Instance 2b: A[0:2, 2:4] * B[2:4, 2:4] -> C[0:2, 2:4]
    # A indices: same as 2a
    # B indices: j in 2,3; k in 2,3. V_idx = j*4 + k
    # C indices: i in 0,1; k in 2,3. W_idx = k*2 + i
    
    x_map_2b = [0*4+2, 0*4+3, 1*4+2, 1*4+3]
    y_map_2b = [2*4+2, 2*4+3, 3*4+2, 3*4+3] # B22, B23, B32, B33
    z_map_2b = [2*2+0, 2*2+1, 3*2+0, 3*2+1]
    
    add_strassen_instance(21, x_map_2b, y_map_2b, z_map_2b)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
