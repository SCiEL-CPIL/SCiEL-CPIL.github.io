"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based decomposition for 2x4 * 4x8
    # Rank 56: Two independent 2x2x8 Strassen decompositions
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]  # 8 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 16 x 56
    
    # Strassen 2x2 decomposition
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    # c00 = M1 + M4 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M3 + M6
    
    # We have two blocks of 2x2 A matrices (A_L, A_R) and corresponding 2x8 B matrices (B_T, B_B).
    # Actually, we split A into two 2x2 matrices A1, A2.
    # We split B into two 2x8 matrices B1, B2.
    # C = A1 B1 + A2 B2.
    
    # For each 2x2x8 block, we need 28 multiplications.
    # Let's define the Strassen coefficients for one 2x2x2 block.
    # Then repeat for the 4 column-pairs in the 8 columns.
    
    # Indices for A (2x4): i in 0..1, j in 0..3
    # Indices for B (4x8): j in 0..3, k in 0..7
    # Indices for C (2x8): i in 0..1, k in 0..7
    
    # Block 1: A1 (cols 0,1), B1 (rows 0,1)
    # Block 2: A2 (cols 2,3), B2 (rows 2,3)
    
    # Helper to add a Strassen 2x2x2 decomposition at a specific offset
    # a_idx: list of 4 indices in U (i*m+j)
    # b_idx: list of 4 indices in V (j*p+k) - note: B is 4x8, so row j, col k
    # c_idx: list of 4 indices in W (k*n+i)
    # rank_offset: starting column index
    
    def add_strassen_2x2(a_rows, a_cols, b_rows, b_cols, c_rows, c_cols, offset):
        # a is 2x2, b is 2x2, c is 2x2
        # a00, a01, a10, a11
        # b00, b01, b10, b11
        # c00, c01, c10, c11
        
        # M1 = (a00+a11)(b00+b11)
        U[a_rows[0]*m + a_cols[0]][offset+0] = 1
        U[a_rows[1]*m + a_cols[1]][offset+0] = 1
        V[b_rows[0]*p + b_cols[0]][offset+0] = 1
        V[b_rows[1]*p + b_cols[1]][offset+0] = 1
        W[c_rows[0]*n + c_cols[0]][offset+0] = 1
        W[c_rows[1]*n + c_cols[1]][offset+0] = 1
        W[c_rows[1]*n + c_cols[0]][offset+0] = -1
        W[c_rows[0]*n + c_cols[1]][offset+0] = 1 # c01 part of M1? No, M1 contributes to c00 and c11.
        # Wait, Strassen reconstruction:
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # Let's map the 7 multiplications to columns.
        
        # Col 0: M1
        U[a_rows[0]*m + a_cols[0]][offset+0] = 1
        U[a_rows[1]*m + a_cols[1]][offset+0] = 1
        V[b_rows[0]*p + b_cols[0]][offset+0] = 1
        V[b_rows[1]*p + b_cols[1]][offset+0] = 1
        W[c_rows[0]*n + c_cols[0]][offset+0] = 1
        W[c_rows[1]*n + c_cols[1]][offset+0] = 1

        # Col 1: M2 = (a10+a11)b00
        U[a_rows[1]*m + a_cols[0]][offset+1] = 1
        U[a_rows[1]*m + a_cols[1]][offset+1] = 1
        V[b_rows[0]*p + b_cols[0]][offset+1] = 1
        W[c_rows[1]*n + c_cols[0]][offset+1] = 1
        W[c_rows[1]*n + c_cols[1]][offset+1] = -1

        # Col 2: M3 = a00(b01-b11)
        U[a_rows[0]*m + a_cols[0]][offset+2] = 1
        V[b_rows[0]*p + b_cols[1]][offset+2] = 1
        V[b_rows[1]*p + b_cols[1]][offset+2] = -1
        W[c_rows[0]*n + c_cols[1]][offset+2] = 1
        W[c_rows[1]*n + c_cols[1]][offset+2] = 1

        # Col 3: M4 = a11(b10-b00)
        U[a_rows[1]*m + a_cols[1]][offset+3] = 1
        V[b_rows[1]*p + b_cols[0]][offset+3] = 1
        V[b_rows[0]*p + b_cols[0]][offset+3] = -1
        W[c_rows[0]*n + c_cols[0]][offset+3] = 1
        W[c_rows[1]*n + c_cols[0]][offset+3] = 1

        # Col 4: M5 = (a00+a01)b11
        U[a_rows[0]*m + a_cols[0]][offset+4] = 1
        U[a_rows[0]*m + a_cols[1]][offset+4] = 1
        V[b_rows[1]*p + b_cols[1]][offset+4] = 1
        W[c_rows[0]*n + c_cols[0]][offset+4] = -1
        W[c_rows[0]*n + c_cols[1]][offset+4] = 1

        # Col 5: M6 = (a10-a00)(b00+b10)
        U[a_rows[1]*m + a_cols[0]][offset+5] = 1
        U[a_rows[0]*m + a_cols[0]][offset+5] = -1
        V[b_rows[0]*p + b_cols[0]][offset+5] = 1
        V[b_rows[1]*p + b_cols[0]][offset+5] = 1
        W[c_rows[1]*n + c_cols[1]][offset+5] = 1

        # Col 6: M7 = (a01-a11)(b01+b11)
        U[a_rows[0]*m + a_cols[1]][offset+6] = 1
        U[a_rows[1]*m + a_cols[1]][offset+6] = -1
        V[b_rows[0]*p + b_cols[1]][offset+6] = 1
        V[b_rows[1]*p + b_cols[1]][offset+6] = 1
        W[c_rows[0]*n + c_cols[0]][offset+6] = 1
        W[c_rows[1]*n + c_cols[1]][offset+6] = 1

    # Block 1: A1 (rows 0-1, cols 0-1), B1 (rows 0-1, cols 0-7)
    # A1 indices in U: (0,0), (0,1), (1,0), (1,1) -> 0, 1, 4, 5
    # B1 indices in V: rows 0,1. Cols 0-7.
    # We need to apply Strassen for each 2x2 block of B1.
    # B1 has 4 column pairs: (0,1), (2,3), (4,5), (6,7).
    
    # For each column pair (k, k+1):
    # B1 submatrix is 2x2.
    # a_rows = [0, 1], a_cols = [0, 1]
    # b_rows = [0, 1]
    # b_cols = [k, k+1]
    # c_rows = [0, 1]
    # c_cols = [k, k+1]
    
    offset = 0
    for k in range(0, 8, 2):
        add_strassen_2x2(
            [0, 1], [0, 1],  # A1
            [0, 1], [k, k+1], # B1 block
            [0, 1], [k, k+1], # C block
            offset
        )
        offset += 7

    # Block 2: A2 (rows 0-1, cols 2-3), B2 (rows 2-3, cols 0-7)
    # A2 indices in U: (0,2), (0,3), (1,2), (1,3) -> 2, 3, 6, 7
    # B2 indices in V: rows 2,3. Cols 0-7.
    
    for k in range(0, 8, 2):
        add_strassen_2x2(
            [0, 1], [2, 3],  # A2
            [2, 3], [k, k+1], # B2 block
            [0, 1], [k, k+1], # C block
            offset
        )
        offset += 7

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
