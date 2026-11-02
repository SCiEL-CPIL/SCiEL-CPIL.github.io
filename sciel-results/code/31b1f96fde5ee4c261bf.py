"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 * 7 = 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # Local indices for X (A), Y (B), Z (C) are 0..3 corresponding to (0,0), (0,1), (1,0), (1,1)
    # U_local[r][local_x]
    # V_local[r][local_y]
    # W_local[r][local_z]
    
    # r=0 (M1)
    u0 = [1, 0, 0, 0]
    v0 = [1, 0, 0, 1]
    w0 = [0, 1, 0, 1] # z00=0, z01=1, z10=0, z11=1
    
    # r=1 (M2)
    u1 = [1, 1, 0, 0]
    v1 = [0, 0, 0, 1]
    w1 = [-1, 1, 0, 0] # z00=-1, z01=1, z10=0, z11=0
    
    # r=2 (M3)
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [0, 0, 1, -1] # z00=0, z01=0, z10=1, z11=-1
    
    # r=3 (M4)
    u3 = [0, 0, 0, 1]
    v3 = [-1, 0, 1, 0] # y10 - y00 -> y00=-1, y10=1
    w3 = [1, 0, 1, 0] # z00=1, z01=0, z10=1, z11=0
    
    # r=4 (M5)
    u4 = [1, 0, 0, 1]
    v4 = [1, 0, 0, 1]
    w4 = [1, 0, 0, 1] # z00=1, z01=0, z10=0, z11=1
    
    # r=5 (M6)
    u5 = [0, 1, 0, -1] # x01 - x11
    v5 = [0, 0, 1, 1] # y10 + y11
    w5 = [1, 0, 0, 0] # z00=1
    
    # r=6 (M7)
    u6 = [1, 0, -1, 0] # x00 - x10
    v6 = [1, 0, 1, 0] # y00 + y10
    w6 = [0, 0, 0, -1] # z11=-1
    
    strassen_u = [u0, u1, u2, u3, u4, u5, u6]
    strassen_v = [v0, v1, v2, v3, v4, v5, v6]
    strassen_w = [w0, w1, w2, w3, w4, w5, w6]
    
    # Define mappings for the 4 sub-problems
    # Each sub-problem has:
    # map_x: local_x -> global_A_idx
    # map_y: local_y -> global_B_idx
    # map_z: local_z -> global_C_idx
    
    # Global A idx: i*4 + j
    # Global B idx: j*4 + k
    # Global C idx: k*2 + i
    
    # Sub-problem 0: A_L * B_LL -> C[:, 0:2]
    # A_L: A[0,0], A[0,1], A[1,0], A[1,1] -> idx 0, 1, 4, 5
    # B_LL: B[0,0], B[0,1], B[1,0], B[1,1] -> idx 0, 1, 4, 5
    # C[:, 0:2]: C[0,0], C[0,1], C[1,0], C[1,1] -> idx 0, 2, 1, 3
    sp0 = {
        'map_x': [0, 1, 4, 5],
        'map_y': [0, 1, 4, 5],
        'map_z': [0, 2, 1, 3]
    }
    
    # Sub-problem 1: A_R * B_RL -> C[:, 0:2]
    # A_R: A[0,2], A[0,3], A[1,2], A[1,3] -> idx 2, 3, 6, 7
    # B_RL: B[2,0], B[2,1], B[3,0], B[3,1] -> idx 8, 9, 12, 13
    # C[:, 0:2]: Same as SP0
    sp1 = {
        'map_x': [2, 3, 6, 7],
        'map_y': [8, 9, 12, 13],
        'map_z': [0, 2, 1, 3]
    }
    
    # Sub-problem 2: A_L * B_LR -> C[:, 2:4]
    # A_L: Same as SP0
    # B_LR: B[0,2], B[0,3], B[1,2], B[1,3] -> idx 2, 3, 6, 7
    # C[:, 2:4]: C[0,2], C[0,3], C[1,2], C[1,3] -> idx 4, 6, 5, 7
    sp2 = {
        'map_x': [0, 1, 4, 5],
        'map_y': [2, 3, 6, 7],
        'map_z': [4, 6, 5, 7]
    }
    
    # Sub-problem 3: A_R * B_RR -> C[:, 2:4]
    # A_R: Same as SP1
    # B_RR: B[2,2], B[2,3], B[3,2], B[3,3] -> idx 10, 11, 14, 15
    # C[:, 2:4]: Same as SP2
    sp3 = {
        'map_x': [2, 3, 6, 7],
        'map_y': [10, 11, 14, 15],
        'map_z': [4, 6, 5, 7]
    }
    
    sub_problems = [sp0, sp1, sp2, sp3]
    
    col_offset = 0
    for sp in sub_problems:
        for r in range(7):
            global_col = col_offset + r
            # Fill U
            for lx in range(4):
                val = strassen_u[r][lx]
                if val != 0:
                    U[sp['map_x'][lx]][global_col] = val
            
            # Fill V
            for ly in range(4):
                val = strassen_v[r][ly]
                if val != 0:
                    V[sp['map_y'][ly]][global_col] = val
            
            # Fill W
            for lz in range(4):
                val = strassen_w[r][lz]
                if val != 0:
                    W[sp['map_z'][lz]][global_col] = val
        
        col_offset += 7
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
