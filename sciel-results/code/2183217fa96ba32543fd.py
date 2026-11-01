"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # p is 14, so we have 7 groups of 2 columns
    num_groups = p // 2
    rank = 7 * num_groups
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen multipliers for 2x2 matrix multiplication
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    for g in range(num_groups):
        base_col = g * 7
        # Column indices in B and C for this group
        # B has columns 2g and 2g+1
        # C has columns 2g and 2g+1
        
        # Indices in V (m*p = 2*14 = 28)
        # V index = j * p + k
        v_00 = 0 * p + 2 * g
        v_01 = 0 * p + 2 * g + 1
        v_10 = 1 * p + 2 * g
        v_11 = 1 * p + 2 * g + 1
        
        # Indices in W (p*n = 14*2 = 28)
        # W index = k * n + i
        w_00 = 2 * g * n + 0
        w_01 = (2 * g + 1) * n + 0
        w_10 = 2 * g * n + 1
        w_11 = (2 * g + 1) * n + 1
        
        # Indices in U (n*m = 2*2 = 4)
        # U index = i * m + j
        u_00 = 0
        u_01 = 1
        u_10 = 2
        u_11 = 3
        
        # M1
        r = base_col
        U[u_00][r] = 1
        U[u_11][r] = 1
        V[v_00][r] = 1
        V[v_11][r] = 1
        W[w_00][r] = 1
        W[w_11][r] = 1
        
        # M2
        r = base_col + 1
        U[u_10][r] = 1
        U[u_11][r] = 1
        V[v_00][r] = 1
        W[w_10][r] = 1
        
        # M3
        r = base_col + 2
        U[u_00][r] = 1
        V[v_01][r] = 1
        V[v_11][r] = -1
        W[w_01][r] = 1
        
        # M4
        r = base_col + 3
        U[u_11][r] = 1
        V[v_10][r] = 1
        V[v_00][r] = -1
        W[w_10][r] = 1
        
        # M5
        r = base_col + 4
        U[u_00][r] = 1
        U[u_01][r] = 1
        V[v_11][r] = 1
        W[w_00][r] = -1
        W[w_01][r] = 1
        
        # M6
        r = base_col + 5
        U[u_10][r] = 1
        U[u_00][r] = -1
        V[v_00][r] = 1
        V[v_01][r] = 1
        W[w_11][r] = 1
        
        # M7
        r = base_col + 6
        U[u_01][r] = 1
        U[u_11][r] = -1
        V[v_10][r] = 1
        V[v_11][r] = 1
        W[w_00][r] = 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
