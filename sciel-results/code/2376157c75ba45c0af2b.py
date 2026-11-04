"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using 4 Strassen blocks
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients
    # M1: (a+d)(e+h)
    # M2: (c+d)e
    # M3: a(f-h)
    # M4: d(g-e)
    # M5: (a+b)h
    # M6: (c-a)(e+f)
    # M7: (b-d)(g+h)
    
    # Local indices in 2x2 matrices:
    # A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # Strassen steps: list of (u_coeffs, v_coeffs, w_coeffs)
    # u_coeffs: dict mapping local A index to coeff
    # v_coeffs: dict mapping local B index to coeff
    # w_coeffs: dict mapping local C index to coeff
    
    steps = [
        # M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: -1, 2: 1, 1: -1}), # C11+=M1, C22+=M1
        # M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}), # C21+=M2, C22-=M2
        # M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}), # C12+=M3, C22+=M3
        # M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}), # C11+=M4, C21+=M4
        # M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}), # C11-=M5, C12+=M5
        # M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}), # C22+=M6
        # M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}), # C11+=M7
    ]
    
    # 4 blocks: (j_block, k_block)
    blocks = [(0, 0), (1, 0), (0, 1), (1, 1)]
    
    col = 0
    for j_b, k_b in blocks:
        for u_c, v_c, w_c in steps:
            # Update U
            for local_idx, coeff in u_c.items():
                r, c_local = divmod(local_idx, 2)
                global_j = 2 * j_b + c_local
                global_u = r * m + global_j
                U[global_u][col] += coeff
                
            # Update V
            for local_idx, coeff in v_c.items():
                r, c_local = divmod(local_idx, 2)
                global_j = 2 * j_b + r
                global_k = 2 * k_b + c_local
                global_v = global_j * p + global_k
                V[global_v][col] += coeff
                
            # Update W
            for local_idx, coeff in w_c.items():
                r, c_local = divmod(local_idx, 2)
                global_k = 2 * k_b + c_local
                global_i = r
                global_w = global_k * n + global_i
                W[global_w][col] += coeff
                
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
