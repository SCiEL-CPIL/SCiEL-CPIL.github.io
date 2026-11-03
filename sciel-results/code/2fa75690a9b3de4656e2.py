"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 4 blocks * 7 Strassen terms = 28
    rank = 28
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # terms[s] = (U_coeffs, V_coeffs, W_coeffs)
    terms = [
        # 0: M1 = (x00+x11)(y00+y11) -> z00 += M1, z11 += M1
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        # 1: M2 = (x10+x11)y00 -> z10 += M2, z11 -= M2
        ({2: 1, 3: 1}, {0: 1}, {2: 1, 3: -1}),
        # 2: M3 = x00(y01-y11) -> z01 += M3, z11 += M3
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        # 3: M4 = x11(y10-y00) -> z00 += M4, z10 += M4
        ({3: 1}, {2: 1, 0: -1}, {0: 1, 2: 1}),
        # 4: M5 = (x00+x01)y11 -> z00 -= M5, z01 += M5
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        # 5: M6 = (x10-x00)(y00+y01) -> z11 += M6
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        # 6: M7 = (x01-x11)(y10+y11) -> z00 += M7
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),
    ]
    
    # Block definitions:
    # Block 0: A_L (j=0,1) * B_LL (j=0,1, r=0,1) -> C_L (r=0,1)
    # Block 1: A_R (j=2,3) * B_RL (j=2,3, r=0,1) -> C_L (r=0,1)
    # Block 2: A_L (j=0,1) * B_LR (j=0,1, r=2,3) -> C_R (r=2,3)
    # Block 3: A_R (j=2,3) * B_RR (j=2,3, r=2,3) -> C_R (r=2,3)
    
    def get_global_A(block, li, lj):
        i = li
        if block == 0: # A_L
            j = lj
        elif block == 1: # A_R
            j = 2 + lj
        elif block == 2: # A_L
            j = lj
        elif block == 3: # A_R
            j = 2 + lj
        return i * 4 + j
        
    def get_global_B(block, li, lj):
        j_local = li
        r_local = lj
        if block == 0: # B_LL
            j = j_local
            r = r_local
        elif block == 1: # B_RL
            j = 2 + j_local
            r = r_local
        elif block == 2: # B_LR
            j = j_local
            r = 2 + r_local
        elif block == 3: # B_RR
            j = 2 + j_local
            r = 2 + r_local
        return j * 4 + r
        
    def get_global_C(block, li, lj):
        i_local = li
        r_local = lj
        if block == 0: # C_L
            i = i_local
            r = r_local
        elif block == 1: # C_L
            i = i_local
            r = r_local
        elif block == 2: # C_R
            i = i_local
            r = 2 + r_local
        elif block == 3: # C_R
            i = i_local
            r = 2 + r_local
        return r * 2 + i

    # Iterate blocks
    for b in range(4):
        for s in range(7):
            col = b * 7 + s
            u_coeffs, v_coeffs, w_coeffs = terms[s]
            
            for local_idx, coeff in u_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                g_idx = get_global_A(b, li, lj)
                U[g_idx][col] += coeff
                
            for local_idx, coeff in v_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                g_idx = get_global_B(b, li, lj)
                V[g_idx][col] += coeff
                
            for local_idx, coeff in w_coeffs.items():
                li = local_idx // 2
                lj = local_idx % 2
                g_idx = get_global_C(b, li, lj)
                W[g_idx][col] += coeff

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
