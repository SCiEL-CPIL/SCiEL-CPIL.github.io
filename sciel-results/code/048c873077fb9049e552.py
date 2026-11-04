"""Decomposition for 4x4x9 matrix multiplication using block structure."""
DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START

def _decompose_2x2x3():
    """
    Rank-11 decomposition for 2x2 times 2x3 matrix multiplication.
    A is 2x2 (indices 0-3), B is 2x3 (indices 0-5), C is 2x3 (indices 0-5).
    Returns (U, V, W) where U is 4x11, V is 6x11, W is 6x11.
    U[a_index, mult_index] = coefficient for A element in multiplication
    V[b_index, mult_index] = coefficient for B element in multiplication
    W[c_index, mult_index] = coefficient for C element (output)
    """
    rank = 11
    U = [[0.0] * rank for _ in range(4)]
    V = [[0.0] * rank for _ in range(6)]
    W = [[0.0] * rank for _ in range(6)]
    
    # M1 = (A0+A3)(B0+B3)
    U[0][0] = 1.0; U[3][0] = 1.0
    V[0][0] = 1.0; V[3][0] = 1.0
    # M2 = (A2+A3)B0
    U[2][1] = 1.0; U[3][1] = 1.0
    V[0][1] = 1.0
    # M3 = A0(B1-B3)
    U[0][2] = 1.0
    V[1][2] = 1.0; V[3][2] = -1.0
    # M4 = A3(B2-B0)
    U[3][3] = 1.0
    V[2][3] = 1.0; V[0][3] = -1.0
    # M5 = (A0+A1)B3
    U[0][4] = 1.0; U[1][4] = 1.0
    V[3][4] = 1.0
    # M6 = (A2-A0)(B0+B1)
    U[2][5] = 1.0; U[0][5] = -1.0
    V[0][5] = 1.0; V[1][5] = 1.0
    # M7 = (A1-A3)(B2+B3)
    U[1][6] = 1.0; U[3][6] = -1.0
    V[2][6] = 1.0; V[3][6] = 1.0
    # M8 = A0*B4
    U[0][7] = 1.0
    V[4][7] = 1.0
    # M9 = A1*B4
    U[1][8] = 1.0
    V[4][8] = 1.0
    # M10 = A2*B5
    U[2][9] = 1.0
    V[5][9] = 1.0
    # M11 = A3*B5
    U[3][10] = 1.0
    V[5][10] = 1.0
    
    # C = A * B
    # C[0] = A0*B0 + A1*B2
    # C[1] = A0*B1 + A1*B3
    # C[2] = A2*B0 + A3*B2
    # C[3] = A2*B1 + A3*B3
    # C[4] = A0*B4 + A1*B5
    # C[5] = A2*B4 + A3*B5
    
    # C0 = M1 + M4 - M5
    W[0][0] = 1.0; W[0][3] = 1.0; W[0][4] = -1.0
    # C1 = M3 + M5
    W[1][2] = 1.0; W[1][4] = 1.0
    # C2 = M2 + M4
    W[2][1] = 1.0; W[2][3] = 1.0
    # C3 = M1 - M2 + M3 + M6
    W[3][0] = 1.0; W[3][1] = -1.0; W[3][2] = 1.0; W[3][5] = 1.0
    # C4 = M8 + M9
    W[4][7] = 1.0; W[4][8] = 1.0
    # C5 = M10 + M11
    W[5][9] = 1.0; W[5][10] = 1.0
    
    return U, V, W


def _decompose_4x4x3():
    """
    Rank-44 decomposition for 4x4 times 4x3 matrix multiplication.
    A is 4x4 (16 elements), B is 4x3 (12 elements), C is 4x3 (12 elements).
    Returns (U, V, W) where U is 16x44, V is 12x44, W is 12x44.
    """
    u2, v2, w2 = _decompose_2x2x3()
    rank = 44
    
    U = [[0.0] * rank for _ in range(16)]
    V = [[0.0] * rank for _ in range(12)]
    W = [[0.0] * rank for _ in range(12)]
    
    # Sub-problem (row_block, col_block)
    # A indices: for row_block r, col_block c: [r*4 + c*2, r*4 + c*2 + 1, (r+2)*4 + c*2, (r+2)*4 + c*2 + 1]
    # B indices: for col_block c: [c*2, c*2+1, (c+2)*2, (c+2)*2+1, c*2+2, (c+2)*2+2]
    # C indices: for row_block r, col_block c: [r*2 + c, r*2 + c + 1, (r+2)*2 + c, (r+2)*2 + c + 1]
    
    for r_block in range(2):
        for c_block in range(2):
            offset = (r_block * 2 + c_block) * 11
            
            # A element mapping: local (0,1,2,3) -> global
            a_map = [
                r_block * 4 + c_block * 2,
                r_block * 4 + c_block * 2 + 1,
                (r_block + 2) * 4 + c_block * 2,
                (r_block + 2) * 4 + c_block * 2 + 1
            ]
            
            # B element mapping: local (0,1,2,3,4,5) -> global
            b_map = [
                c_block * 2,
                c_block * 2 + 1,
                (c_block + 2) * 2,
                (c_block + 2) * 2 + 1,
                c_block * 2 + 2,
                (c_block + 2) * 2 + 2
            ]
            
            # C element mapping: local (0,1,2,3,4,5) -> global
            c_map = [
                r_block * 2 + c_block,
                r_block * 2 + c_block + 1,
                (r_block + 2) * 2 + c_block,
                (r_block + 2) * 2 + c_block + 1,
                r_block * 2 + c_block + 2,
                (r_block + 2) * 2 + c_block + 2
            ]
            
            for mult_idx in range(11):
                col = offset + mult_idx
                
                # Fill U
                for local_a, coeff in [(0, u2[0][mult_idx]), (1, u2[1][mult_idx]), 
                                        (2, u2[2][mult_idx]), (3, u2[3][mult_idx])]:
                    if coeff != 0:
                        U[a_map[local_a]][col] += coeff
                
                # Fill V
                for local_b, coeff in [(0, v2[0][mult_idx]), (1, v2[1][mult_idx]),
                                        (2, v2[2][mult_idx]), (3, v2[3][mult_idx]),
                                        (4, v2[4][mult_idx]), (5, v2[5][mult_idx])]:
                    if coeff != 0:
                        V[b_map[local_b]][col] += coeff
                
                # Fill W
                for local_c, coeff in [(0, w2[0][mult_idx]), (1, w2[1][mult_idx]),
                                        (2, w2[2][mult_idx]), (3, w2[3][mult_idx]),
                                        (4, w2[4][mult_idx]), (5, w2[5][mult_idx])]:
                    if coeff != 0:
                        W[c_map[local_c]][col] += coeff
    
    return U, V, W


def construct():
    """
    Construct rank-132 decomposition for 4x4 x 4x9 matrix multiplication.
    U: (16, 132), V: (36, 132), W: (36, 132)
    W uses trace convention: W[k*n+i, c] where n=4, so W[k*4+i, c]
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=9
    
    u4, v4, w4 = _decompose_4x4x3()
    rank = 132
    
    U = [[0.0] * rank for _ in range(n * m)]  # 16 x 132
    V = [[0.0] * rank for _ in range(m * p)]  # 36 x 132
    W = [[0.0] * rank for _ in range(p * n)]  # 36 x 132
    
    # Three blocks of 4x3
    for block in range(3):
        offset = block * 44
        
        # U: same A indices for each block
        for i in range(16):
            for c in range(44):
                if u4[i][c] != 0:
                    U[i][offset + c] = u4[i][c]
        
        # V: B column offset = block * 3
        # B is 4x9, so B[j, k] where j in 0..3 (rows), k in 0..8 (cols)
        # V row index = j * p + k = j * 9 + k
        # For block b, columns k = b*3, b*3+1, b*3+2
        # In the 4x3 decomposition, B local index: row r (0..3), col c_local (0..2) -> global V index = r*9 + (b*3 + c_local)
        # In v4, V has 12 rows: index = r*3 + c_local for r in 0..3, c_local in 0..2
        for local_v_row in range(12):
            r = local_v_row // 3
            c_local = local_v_row % 3
            global_v_row = r * p + (block * 3 + c_local)
            for c in range(44):
                if v4[local_v_row][c] != 0:
                    V[global_v_row][offset + c] = v4[local_v_row][c]
        
        # W: C column offset = block * 3
        # W uses trace convention: W[k*n + i, c] where k is col of C (0..8), i is row of C (0..3)
        # In the 4x3 decomposition, C has 12 elements: row r (0..3), col c_local (0..2) -> index = r*3 + c_local
        # Global W index for C[r, block*3 + c_local] = (block*3 + c_local)*n + r
        for local_w_row in range(12):
            r = local_w_row // 3
            c_local = local_w_row % 3
            global_w_row = (block * 3 + c_local) * n + r
            for c in range(44):
                if w4[local_w_row][c] != 0:
                    W[global_w_row][offset + c] = w4[local_w_row][c]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
