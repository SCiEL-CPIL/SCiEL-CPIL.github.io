"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We use a Strassen-based decomposition.
    # The 2x6 matrix is viewed as three 2x2 blocks.
    # Each 2x2 multiplication has rank 7 (Strassen).
    # Total rank = 3 * 7 = 21.
    
    rank = 21
    U = [[0] * rank for _ in range(n * m)] # 4 x 21
    V = [[0] * rank for _ in range(m * p)] # 12 x 21
    W = [[0] * rank for _ in range(p * n)] # 12 x 21
    
    # Strassen coefficients for 2x2 matrix multiplication
    # A = [[a11, a12], [a21, a22]]
    # B = [[b11, b12], [b21, b22]]
    # Indices for A: 0:a11, 1:a12, 2:a21, 3:a22
    # Indices for B: 0:b11, 1:b12, 2:b21, 3:b22
    # Indices for C: 0:c11, 1:c12, 2:c21, 3:c22
    
    # Strassen products M1..M7
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # U coefficients (for A)
    U_M1 = [1, 0, 0, 1]
    U_M2 = [0, 0, 1, 1]
    U_M3 = [1, 0, 0, 0]
    U_M4 = [0, 0, 0, 1]
    U_M5 = [1, 1, 0, 0]
    U_M6 = [-1, 0, 1, 0]
    U_M7 = [0, 1, 0, -1]
    
    # V coefficients (for B)
    V_M1 = [1, 0, 0, 1]
    V_M2 = [1, 0, 0, 0]
    V_M3 = [0, 1, 0, -1]
    V_M4 = [0, 0, 1, -1]
    V_M5 = [0, 0, 0, 1]
    V_M6 = [1, 1, 0, 0]
    V_M7 = [0, 0, 1, 1]
    
    # W coefficients (for C)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # W[row, col] where row is C index, col is M index
    # We need to map these to the global W matrix.
    
    U_strassen = [U_M1, U_M2, U_M3, U_M4, U_M5, U_M6, U_M7]
    V_strassen = [V_M1, V_M2, V_M3, V_M4, V_M5, V_M6, V_M7]
    
    # W_strassen[m_idx][c_idx]
    W_strassen = [
        [1, 0, 0, 1, -1, 0, 1],   # C11
        [0, 0, 1, 0, 1, 0, 0],    # C12
        [0, 1, 0, 1, 0, 0, 0],    # C21
        [1, -1, 1, 0, 0, 1, 0]    # C22
    ]
    
    # Fill U, V, W for 3 blocks
    for block in range(3):
        base_col = block * 7
        base_v = block * 4
        base_w = block * 4
        
        for k in range(7):
            col = base_col + k
            # Fill U
            for i in range(4):
                U[i][col] = U_strassen[k][i]
            
            # Fill V
            for j in range(4):
                V[base_v + j][col] = V_strassen[k][j]
            
            # Fill W
            for c_idx in range(4):
                W[base_w + c_idx][col] = W_strassen[c_idx][k]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
