"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition
    rank = 56
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)] # 8 x 56
    V = [[0] * rank for _ in range(m * p)] # 32 x 56
    W = [[0] * rank for _ in range(p * n)] # 16 x 56
    
    # Strassen factors for 2x2
    # A indices: 0:a, 1:b, 2:c, 3:d
    # B indices: 0:e, 1:f, 2:g, 3:h
    # C indices: 0:C11, 1:C12, 2:C21, 3:C22
    
    U_strassen = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    V_strassen = [
        [1, 1, 0, 0], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    W_strassen = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # A is 2x4. Split into A_L (cols 0,1) and A_R (cols 2,3).
    # A_L indices in U: 0,1,2,3 (a,b,c,d)
    # A_R indices in U: 4,5,6,7 (a,b,c,d)
    
    # B is 4x8. Split into 4 blocks B_j (4x2).
    # B_j has rows 0,1 (B_T) and 2,3 (B_B).
    # B_T indices in V: 4*j, 4*j+1, 4*j+2, 4*j+3? No.
    # B is 4x8. Row-major.
    # B[row, col].
    # Block j uses cols 2*j, 2*j+1.
    # B_T,j uses rows 0,1 and cols 2*j, 2*j+1.
    # Indices: 0*4 + 2*j, 0*4 + 2*j+1, 1*4 + 2*j, 1*4 + 2*j+1.
    # B_B,j uses rows 2,3 and cols 2*j, 2*j+1.
    # Indices: 2*4 + 2*j, 2*4 + 2*j+1, 3*4 + 2*j, 3*4 + 2*j+1.
    
    # C is 2x8. Split into 4 blocks C_j (2x2).
    # C_j uses cols 2*j, 2*j+1.
    # C_T,j uses rows 0,1.
    # Indices: 0*8 + 2*j, 0*8 + 2*j+1, 1*8 + 2*j, 1*8 + 2*j+1.
    
    for j in range(4):
        # Block j starts at column 14*j
        col_offset = 14 * j
        
        # First 7 multiplications: A_L * B_T,j
        # U: A_L (indices 0-3)
        # V: B_T,j (indices 4*j, 4*j+1, 4*j+2, 4*j+3)
        # W: C_T,j (indices 2*j, 2*j+1, 8+2*j, 8+2*j+1)
        
        for k in range(7):
            c = col_offset + k
            # U
            for i in range(4):
                U[i][c] = U_strassen[k][i]
            # V
            v_base = 4 * j
            for i in range(4):
                V[v_base + i][c] = V_strassen[k][i]
            # W
            w_base = 2 * j
            # C indices: 0,1 (row 0), 8,9 (row 1) relative to block?
            # C is 2x8. Row 0: 0..7. Row 1: 8..15.
            # Block j cols: 2j, 2j+1.
            # C11: 2j. C12: 2j+1.
            # C21: 8+2j. C22: 8+2j+1.
            W[w_base][c] = W_strassen[k][0]
            W[w_base + 1][c] = W_strassen[k][1]
            W[8 + w_base][c] = W_strassen[k][2]
            W[8 + w_base + 1][c] = W_strassen[k][3]
            
        # Next 7 multiplications: A_R * B_B,j
        # U: A_R (indices 4-7)
        # V: B_B,j (indices 8+4*j, 8+4*j+1, 8+4*j+2, 8+4*j+3)
        # W: C_T,j (same as above, add to W)
        
        for k in range(7):
            c = col_offset + 7 + k
            # U
            for i in range(4):
                U[4 + i][c] = U_strassen[k][i]
            # V
            v_base = 8 + 4 * j
            for i in range(4):
                V[v_base + i][c] = V_strassen[k][i]
            # W
            w_base = 2 * j
            W[w_base][c] += W_strassen[k][0]
            W[w_base + 1][c] += W_strassen[k][1]
            W[8 + w_base][c] += W_strassen[k][2]
            W[8 + w_base + 1][c] += W_strassen[k][3]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
