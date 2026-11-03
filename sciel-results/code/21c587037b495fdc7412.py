# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 blocks * 7 Strassen terms = 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # U, V, W coeffs for each of the 7 terms
    # Indices 0..3 map to (0,0), (0,1), (1,0), (1,1)
    strassen_terms = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Block 0: A_L (j=0,1) * B_LL (j=0,1, r=0,1) -> C_L (r=0,1)
    # Block 1: A_R (j=2,3) * B_RL (j=2,3, r=0,1) -> C_L (r=0,1)
    # Block 2: A_L (j=0,1) * B_LR (j=0,1, r=2,3) -> C_R (r=2,3)
    # Block 3: A_R (j=2,3) * B_RR (j=2,3, r=2,3) -> C_R (r=2,3)
    
    for b in range(4):
        for s in range(7):
            col = b * 7 + s
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[s]
            
            for local_idx, coeff in u_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                # A index: i * 4 + j
                # i = li
                # j depends on block
                if b == 0 or b == 2: # A_L
                    j = lj
                else: # A_R
                    j = 2 + lj
                g_idx = li * 4 + j
                U[g_idx][col] += coeff
                
            for local_idx, coeff in v_coeffs.items():
                li = local_idx // 2 # j_local
                lj = local_idx % 2 # r_local
                # B index: j * 4 + r
                # j depends on block
                # r depends on block
                if b == 0 or b == 1: # B_L (cols 0,1)
                    r = lj
                else: # B_R (cols 2,3)
                    r = 2 + lj
                    
                if b == 0 or b == 2: # B_T (rows 0,1)
                    j = li
                else: # B_B (rows 2,3)
                    j = 2 + li
                    
                g_idx = j * 4 + r
                V[g_idx][col] += coeff
                
            for local_idx, coeff in w_coeffs.items():
                li = local_idx // 2 # i_local
                lj = local_idx % 2 # r_local
                # C index: r * 2 + i
                # i = li
                # r depends on block
                if b == 0 or b == 1: # C_L (cols 0,1)
                    r = lj
                else: # C_R (cols 2,3)
                    r = 2 + lj
                    
                g_idx = r * 2 + li
                W[g_idx][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
