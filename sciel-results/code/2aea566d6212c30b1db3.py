"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 33
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 11 terms for 2x3x2 block
    # A: 0=a00, 1=a01, 2=a02, 3=a10, 4=a11, 5=a12
    # B: 0=b00, 1=b01, 2=b10, 3=b11, 4=b20, 5=b21
    # C: 0=c00, 1=c01, 2=c10, 3=c11
    
    terms = [
        # 1
        ({0: 1, 4: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # 2
        ({3: 1, 4: 1}, {0: 1}, {2: 1, 3: -1}),
        # 3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # 4
        ({4: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # 5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # 6
        ({3: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # 7
        ({1: 1, 4: -1}, {2: 1, 3: 1}, {0: 1}),
        # 8
        ({2: 1}, {4: 1}, {0: 1}),
        # 9
        ({2: 1}, {5: 1}, {1: 1}),
        # 10
        ({5: 1}, {4: 1}, {2: 1}),
        # 11
        ({5: 1}, {5: 1}, {3: 1}),
    ]
    
    for block in range(3):
        for t_idx, (u_dict, v_dict, w_dict) in enumerate(terms):
            col = block * 11 + t_idx
            
            for idx, val in u_dict.items():
                U[idx][col] = val
            
            for l_idx, val in v_dict.items():
                j_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = j_local * p + (block * 2 + k_local)
                V[g_idx][col] = val
            
            for l_idx, val in w_dict.items():
                i_local = l_idx // 2
                k_local = l_idx % 2
                g_idx = (block * 2 + k_local) * n + i_local
                W[g_idx][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
