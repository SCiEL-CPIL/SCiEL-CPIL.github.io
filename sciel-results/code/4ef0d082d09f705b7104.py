"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank 14 decomposition using two Strassen 2x2x2 blocks
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # U is 4x7, V is 4x7, W is 4x7
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # C indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    
    # M1 = (A11 - A10)(B00 + B01)
    U_s = [
        [0, 1, 0, 0, 0, 0, 0], # A00
        [0, 0, 0, 0, 0, 0, 0], # A01
        [-1, 0, 0, 0, 0, 0, 0], # A10
        [1, 0, 0, 0, 0, 0, 0]  # A11
    ]
    V_s = [
        [1, 0, 0, 0, 0, 0, 0], # B00
        [1, 0, 0, 0, 0, 0, 0], # B01
        [0, 0, 0, 0, 0, 0, 0], # B10
        [0, 0, 0, 0, 0, 0, 0]  # B11
    ]
    W_s = [
        [0, 1, 0, 0, 0, 0, 0], # C00
        [1, 0, 0, 0, 0, 0, 0], # C01
        [0, 0, 1, 0, 0, 0, 0], # C10
        [0, 0, 0, 1, 0, 0, 0]  # C11
    ]
    
    # M2 = (A00 + A10)B00
    U_s[0][1] = 1
    U_s[2][1] = 1
    V_s[0][1] = 1
    
    # M3 = (A01 + A11)B11
    U_s[1][2] = 1
    U_s[3][2] = 1
    V_s[3][2] = 1
    
    # M4 = A00(B11 - B10)
    U_s[0][3] = 1
    V_s[3][3] = 1
    V_s[2][3] = -1
    
    # M5 = A11(B00 + B10)
    U_s[3][4] = 1
    V_s[0][4] = 1
    V_s[2][4] = 1
    
    # M6 = (A00 - A11)(B01 + B11)
    U_s[0][5] = 1
    U_s[3][5] = -1
    V_s[1][5] = 1
    V_s[3][5] = 1
    
    # M7 = (A10 - A00)(B00 + B01)
    U_s[2][6] = 1
    U_s[0][6] = -1
    V_s[0][6] = 1
    V_s[1][6] = 1
    
    # W coefficients
    # C00 = M5 + M4 - M2 + M6
    W_s[0][4] = 1
    W_s[0][3] = 1
    W_s[0][1] = -1
    W_s[0][5] = 1
    
    # C01 = M1 + M2 - M7 - M5
    W_s[1][0] = 1
    W_s[1][1] = 1
    W_s[1][6] = -1
    W_s[1][4] = -1
    
    # C10 = M3 + M7 - M1 - M4
    W_s[2][2] = 1
    W_s[2][6] = 1
    W_s[2][0] = -1
    W_s[2][3] = -1
    
    # C11 = M5 + M1 - M3 - M6
    W_s[3][4] = 1
    W_s[3][0] = 1
    W_s[3][2] = -1
    W_s[3][5] = -1

    # Fill global U, V, W
    # Block 1: B cols 0, 1 -> V rows 0, 1, 2, 3 (j=0, k=0,1)
    # Block 2: B cols 2, 3 -> V rows 4, 5, 6, 7 (j=0, k=2,3)
    
    for l in range(7):
        # U is shared
        for i in range(4):
            U[i][l] = U_s[i][l]
            U[i][l + 7] = U_s[i][l]
            
        # V
        # Block 1 (cols 0,1)
        for j in range(2):
            for k in range(2):
                v_idx = j * 2 + k
                U_idx = j * 2 + k # Same indexing for 2x2
                V[v_idx][l] = V_s[U_idx][l]
                
        # Block 2 (cols 2,3)
        for j in range(2):
            for k in range(2):
                v_idx = j * 2 + (k + 2)
                U_idx = j * 2 + k
                V[v_idx][l + 7] = V_s[U_idx][l]
                
        # W
        # Block 1 (cols 0,1)
        for i in range(2):
            for k in range(2):
                w_idx = k * 2 + i
                W[w_idx][l] = W_s[w_idx][l]
                
        # Block 2 (cols 2,3)
        for i in range(2):
            for k in range(2):
                w_idx = (k + 2) * 2 + i
                W[w_idx][l + 7] = W_s[w_idx][l]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
