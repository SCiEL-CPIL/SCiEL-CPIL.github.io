"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using two Strassen-like M_{2,2,8} blocks
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrices
    # u: coefficients for A (flattened [A11, A12, A21, A22])
    # v: coefficients for B (flattened [B11, B12, B21, B22])
    # w: coefficients for C (flattened [C11, C12, C21, C22]), each is a vector of 7 coeffs for M1..M7
    
    u = [
        [1, 0, 0, 1],  # M1: A11+A22
        [0, 0, 1, 1],  # M2: A21+A22
        [1, 0, 0, 0],  # M3: A11
        [0, 0, 0, 1],  # M4: A22
        [1, 1, 0, 0],  # M5: A11+A12
        [-1, 0, 1, 0], # M6: A21-A11
        [0, 1, 0, -1]  # M7: A12-A22
    ]
    
    v = [
        [1, 0, 0, 1],  # M1: B11+B22
        [1, 0, 0, 0],  # M2: B11
        [0, 1, 0, -1], # M3: B12-B22
        [0, 0, 1, -1], # M4: B21-B11
        [0, 0, 0, 1],  # M5: B22
        [1, 1, 0, 0],  # M6: B11+B12
        [0, 0, 1, 1]   # M7: B21+B22
    ]
    
    # w[pos][s] is the coefficient of M_s in C_pos
    # C0=C11, C1=C12, C2=C21, C3=C22
    w = [
        [1, 0, 0, 1, -1, 0, 1],  # C11 = M1 + M4 - M5 + M7
        [0, 0, 1, 0, 1, 0, 0],   # C12 = M3 + M5
        [0, 1, 0, 1, 0, 0, 0],   # C21 = M2 + M4
        [1, -1, 1, 0, 0, 1, 0]   # C22 = M1 - M2 + M3 + M6
    ]
    
    # Two sub-problems:
    # sp=0: A_L (cols 0,1), B_top (rows 0,1)
    # sp=1: A_R (cols 2,3), B_bot (rows 2,3)
    
    for sp in range(2):
        # A indices
        if sp == 0:
            a_indices = [0, 1, 4, 5] # A11, A12, A21, A22
        else:
            a_indices = [2, 3, 6, 7]
            
        # B row offset
        b_row_offset = sp * 2
        
        # Column offset in rank
        col_offset = sp * 28
        
        for k in range(4): # 4 blocks of 2x2 in B (and C)
            # B indices for this block
            # Block k corresponds to cols 2k, 2k+1
            b_indices = [
                b_row_offset * 8 + 2*k,       # B11
                b_row_offset * 8 + 2*k + 1,   # B12
                (b_row_offset + 1) * 8 + 2*k,     # B21
                (b_row_offset + 1) * 8 + 2*k + 1  # B22
            ]
            
            # C indices for this block
            c_indices = [
                2*k,       # C11 (row 0)
                2*k + 1,   # C12 (row 0)
                8 + 2*k,       # C21 (row 1)
                8 + 2*k + 1    # C22 (row 1)
            ]
            
            for s in range(7):
                col = col_offset + k * 7 + s
                
                # Fill U
                for a_pos in range(4):
                    val = u[s][a_pos]
                    if val != 0:
                        U[a_indices[a_pos]][col] = val
                        
                # Fill V
                for b_pos in range(4):
                    val = v[s][b_pos]
                    if val != 0:
                        V[b_indices[b_pos]][col] = val
                        
                # Fill W
                for c_pos in range(4):
                    val = w[c_pos][s]
                    if val != 0:
                        W[c_indices[c_pos]][col] = val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
