"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using 4 Strassen multiplications
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Helper to add a Strassen block
    # i_offsets: [i0, i1] for A rows
    # j_offsets: [j0, j1] for A cols (and B rows)
    # k_offsets: [k0, k1] for B cols (and C cols)
    # base_col: starting column index in U, V, W
    def add_strassen(i0, i1, j0, j1, k0, k1, base_col):
        # Strassen products M1..M7
        # M1 = (Y00 + Y11)(Z00 + Z11)
        # M2 = (Y10 + Y11) Z00
        # M3 = Y00 (Z10 - Z11)
        # M4 = Y11 (Z01 - Z00)
        # M5 = (Y00 + Y10) Z11
        # M6 = (Y11 - Y00)(Z00 + Z10)
        # M7 = (Y01 - Y11)(Z11 + Z01)
        
        # Output reconstruction:
        # X00 = M1 + M4 - M5 + M7
        # X01 = M3 + M5
        # X10 = M2 + M4
        # X11 = M1 - M2 + M3 + M6
        
        # We define the contributions to U, V, W for each M
        # U corresponds to Y (A), V to Z (B), W to X (C)
        
        # M1
        U[i0*m + j0][base_col + 0] += 1
        U[i1*m + j1][base_col + 0] += 1
        V[j0*p + k0][base_col + 0] += 1
        V[j1*p + k1][base_col + 0] += 1
        # W for M1 contributes to X00 and X11
        W[k0*n + i0][base_col + 0] += 1
        W[k1*n + i1][base_col + 0] += 1
        
        # M2
        U[i1*m + j0][base_col + 1] += 1
        U[i1*m + j1][base_col + 1] += 1
        V[j0*p + k0][base_col + 1] += 1
        # W for M2 contributes to X10 and X11
        W[k0*n + i1][base_col + 1] += 1
        W[k1*n + i1][base_col + 1] += -1
        
        # M3
        U[i0*m + j0][base_col + 2] += 1
        V[j1*p + k0][base_col + 2] += 1
        V[j1*p + k1][base_col + 2] += -1
        # W for M3 contributes to X01 and X11
        W[k0*n + i0][base_col + 2] += 1
        W[k1*n + i1][base_col + 2] += 1
        
        # M4
        U[i1*m + j1][base_col + 3] += 1
        V[j0*p + k1][base_col + 3] += 1
        V[j0*p + k0][base_col + 3] += -1
        # W for M4 contributes to X00 and X10
        W[k0*n + i0][base_col + 3] += 1
        W[k0*n + i1][base_col + 3] += 1
        
        # M5
        U[i0*m + j0][base_col + 4] += 1
        U[i1*m + j0][base_col + 4] += 1
        V[j1*p + k1][base_col + 4] += 1
        # W for M5 contributes to X00 and X01
        W[k0*n + i0][base_col + 4] += -1
        W[k1*n + i0][base_col + 4] += 1
        
        # M6
        U[i1*m + j1][base_col + 5] += 1
        U[i0*m + j0][base_col + 5] += -1
        V[j0*p + k0][base_col + 5] += 1
        V[j1*p + k0][base_col + 5] += 1
        # W for M6 contributes to X11
        W[k1*n + i1][base_col + 5] += 1
        
        # M7
        U[i0*m + j1][base_col + 6] += 1
        U[i1*m + j1][base_col + 6] += -1
        V[j1*p + k1][base_col + 6] += 1
        V[j0*p + k1][base_col + 6] += 1
        # W for M7 contributes to X00
        W[k0*n + i0][base_col + 6] += 1
        
    # Block 1: A_L B_LL -> C_LL
    # A_L: rows 0,1; cols 0,1
    # B_LL: rows 0,1; cols 0,1
    # C_LL: rows 0,1; cols 0,1
    add_strassen(0, 1, 0, 1, 0, 1, 0)
    
    # Block 2: A_R B_RL -> C_RL (adds to C_L)
    # A_R: rows 0,1; cols 2,3
    # B_RL: rows 2,3; cols 0,1
    # C_RL: rows 0,1; cols 0,1
    add_strassen(0, 1, 2, 3, 0, 1, 7)
    
    # Block 3: A_L B_LR -> C_LR (adds to C_R)
    # A_L: rows 0,1; cols 0,1
    # B_LR: rows 0,1; cols 2,3
    # C_LR: rows 0,1; cols 2,3
    add_strassen(0, 1, 0, 1, 2, 3, 14)
    
    # Block 4: A_R B_RR -> C_RR (adds to C_R)
    # A_R: rows 0,1; cols 2,3
    # B_RR: rows 2,3; cols 2,3
    # C_RR: rows 0,1; cols 2,3
    add_strassen(0, 1, 2, 3, 2, 3, 21)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
