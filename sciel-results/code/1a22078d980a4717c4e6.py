"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 templates
    # Local x indices: 0=x00, 1=x01, 2=x10, 3=x11
    # Local y indices: 0=y00, 1=y01, 2=y10, 3=y11
    # Local z indices: 0=z00, 1=z01, 2=z10, 3=z11
    
    strassen_templates = [
        # 1
        {0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1},
        # 2
        {2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1},
        # 3
        {0: 1}, {2: 1, 0: -1}, {1: 1, 3: 1},
        # 4
        {3: 1}, {3: 1, 0: -1}, {0: 1, 2: 1},
        # 5
        {0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1},
        # 6
        {2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1},
        # 7
        {3: 1, 1: -1}, {1: 1, 3: 1}, {0: 1},
    ]
    
    term_idx = 0
    for j_block in range(3):
        # Block j_block corresponds to B columns 2*j_block and 2*j_block+1
        # And C columns 2*j_block and 2*j_block+1
        
        # Mapping for C (W) indices:
        # z00 -> C[0, 2*j_block] -> W index (2*j_block)*2 + 0
        # z01 -> C[0, 2*j_block+1] -> W index (2*j_block+1)*2 + 0
        # z10 -> C[1, 2*j_block] -> W index (2*j_block)*2 + 1
        # z11 -> C[1, 2*j_block+1] -> W index (2*j_block+1)*2 + 1
        w_map = {
            0: 4 * j_block,
            1: 4 * j_block + 2,
            2: 4 * j_block + 1,
            3: 4 * j_block + 3
        }
        
        # Part 1: A1 * M1_jb
        # A1 is A[:, 0:2]. U indices 0,1,2,3 correspond to x00,x01,x10,x11
        u_map_1 = {0: 0, 1: 1, 2: 2, 3: 3}
        # M1_jb is B[0:2, 2*j_block:2*j_block+2]
        # V indices:
        # y00 -> B[0, 2*j_block] -> 0*6 + 2*j_block
        # y01 -> B[0, 2*j_block+1] -> 0*6 + 2*j_block + 1
        # y10 -> B[1, 2*j_block] -> 1*6 + 2*j_block
        # y11 -> B[1, 2*j_block+1] -> 1*6 + 2*j_block + 1
        v_map_1 = {
            0: 2 * j_block,
            1: 2 * j_block + 1,
            2: 6 + 2 * j_block,
            3: 6 + 2 * j_block + 1
        }
        
        for t in range(7):
            u_coefs, v_coefs, w_coefs = strassen_templates[3*t : 3*t+3]
            for l_idx, val in u_coefs.items():
                g_idx = u_map_1[l_idx]
                U[g_idx][term_idx] += val
            for l_idx, val in v_coefs.items():
                g_idx = v_map_1[l_idx]
                V[g_idx][term_idx] += val
            for l_idx, val in w_coefs.items():
                g_idx = w_map[l_idx]
                W[g_idx][term_idx] += val
            term_idx += 1
            
        # Part 2: A2 * M2_jb
        # A2 is A[:, 2:4]. U indices 4,5,6,7 correspond to x00,x01,x10,x11
        u_map_2 = {0: 4, 1: 5, 2: 6, 3: 7}
        # M2_jb is B[2:4, 2*j_block:2*j_block+2]
        # V indices:
        # y00 -> B[2, 2*j_block] -> 2*6 + 2*j_block
        # y01 -> B[2, 2*j_block+1] -> 2*6 + 2*j_block + 1
        # y10 -> B[3, 2*j_block] -> 3*6 + 2*j_block
        # y11 -> B[3, 2*j_block+1] -> 3*6 + 2*j_block + 1
        v_map_2 = {
            0: 12 + 2 * j_block,
            1: 13 + 2 * j_block,
            2: 18 + 2 * j_block,
            3: 19 + 2 * j_block
        }
        
        for t in range(7):
            u_coefs, v_coefs, w_coefs = strassen_templates[3*t : 3*t+3]
            for l_idx, val in u_coefs.items():
                g_idx = u_map_2[l_idx]
                U[g_idx][term_idx] += val
            for l_idx, val in v_coefs.items():
                g_idx = v_map_2[l_idx]
                V[g_idx][term_idx] += val
            for l_idx, val in w_coefs.items():
                g_idx = w_map[l_idx]
                W[g_idx][term_idx] += val
            term_idx += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
