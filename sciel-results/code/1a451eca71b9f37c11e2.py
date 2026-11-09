"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # We'll build U, V, W matrices
    # U: (n*m) x rank = 12 x rank
    # V: (m*p) x rank = 16 x rank  
    # W: (p*n) x rank = 12 x rank
    
    # Rank 44 decomposition:
    # Part 1: First 2 rows of A (2x4) times B (4x4) using Strassen on 2x2 blocks: rank 28
    # Part 2: 3rd row of A (1x4) times B (4x4) schoolbook: rank 16
    
    entries = []  # list of (u_vec, v_vec, w_vec) tuples
    
    # Part 1: 2x4x4 with Strassen (rank 28)
    # A rows 0,1: indices 0-3 (row0: cols 0-3, row1: cols 0-3)
    # Split A into A_L (cols 0-1) and A_R (cols 2-3)
    # Split B into B00 (rows 0-1, cols 0-1), B01 (rows 0-1, cols 2-3), B10 (rows 2-3, cols 0-1), B11 (rows 2-3, cols 2-3)
    
    def strassen_2x2x2(a_base, b_base, offset):
        """
        Generate 7 rank-1 terms for 2x2x2 Strassen.
        a_base: (row0_col0, row0_col1, row1_col0, row1_col1) indices in U
        b_base: (row0_col0, row0_col1, row1_col0, row1_col1) indices in V
        offset: starting column in W
        Returns list of (u_coeffs, v_coeffs, w_coeffs)
        """
        a00, a01, a10, a11 = a_base
        b00, b01, b10, b11 = b_base
        
        terms = []
        # M1 = (a00+a11)(b00+b11)
        terms.append(([a00, a11], [b00, b11], [offset]))
        # M2 = (a10+a11)*b00
        terms.append(([a10, a11], [b00], [offset+1]))
        # M3 = a00*(b10-b11)
        terms.append(([a00], [b10, b11], [offset+2]))  # b10 - b11
        # M4 = a11*(b01-b00)
        terms.append(([a11], [b00, b01], [offset+3]))  # b01 - b00
        # M5 = (a00+a01)*b11
        terms.append(([a00, a01], [b11], [offset+4]))
        # M6 = (a10-a00)*(b00+b01)
        terms.append(([a00, a10], [b00, b01], [offset+5]))  # (a10-a00)
        # M7 = (a01-a11)*(b10+b11)
        terms.append(([a01, a11], [b10, b11], [offset+6]))  # (a01-a11)
        
        return terms
    
    # U indices: row i of A, col j -> index i*4+j
    # V indices: row j of B, col k -> index j*4+k
    # W indices: row k of C, col i -> index k*3+i (W[k*n+i, r])
    
    # Part 1: rows 0,1 of A
    # A_L: A[0][0], A[0][1], A[1][0], A[1][1] -> U indices 0, 1, 4, 5
    # A_R: A[0][2], A[0][3], A[1][2], A[1][3] -> U indices 2, 3, 6, 7
    # B00: B[0][0], B[0][1], B[1][0], B[1][1] -> V indices 0, 1, 4, 5
    # B01: B[0][2], B[0][3], B[1][2], B[1][3] -> V indices 2, 3, 6, 7
    # B10: B[2][0], B[2][1], B[3][0], B[3][1] -> V indices 8, 9, 12, 13
    # B11: B[2][2], B[2][3], B[3][2], B[3][3] -> V indices 10, 11, 14, 15
    
    # W indices for C: C[k][i] -> W[k*3+i]
    # C[0][0]=W[0], C[0][1]=W[1], C[0][2]=W[2], C[0][3]=W[3]
    # C[1][0]=W[4], C[1][1]=W[5], C[1][2]=W[6], C[1][3]=W[7]
    # C[2][0]=W[8], C[2][1]=W[9], C[2][2]=W[10], C[2][3]=W[11]
    
    # For Strassen 2x2x2, the output C is 2x2:
    # C[0][0], C[0][1], C[1][0], C[1][1]
    # We need to map these to the appropriate W indices
    
    # A_L * B00 contributes to C[0][0], C[0][1], C[1][0], C[1][1] (left half, rows 0-1)
    # Wait, C is 2x4. C_left is 2x2 (cols 0-1), C_right is 2x2 (cols 2-3)
    # C_left[i][j] = C[i][j] for i in 0-1, j in 0-1 -> W[j*3+i]
    # C_right[i][j] = C[i][j+2] for i in 0-1, j in 0-1 -> W[(j+2)*3+i]
    
    # A_L * B00 -> C_left (2x2): output indices W[0], W[1], W[4], W[5] for C[0][0],C[0][1],C[1][0],C[1][1]
    # Actually W[k*n+i] where k is col of C, i is row of C
    # C[0][0] -> W[0*3+0] = W[0]
    # C[0][1] -> W[1*3+0] = W[3]
    # C[1][0] -> W[0*3+1] = W[1]
    # C[1][1] -> W[1*3+1] = W[4]
    
    # Hmm, let me re-examine. W[k*n+i, r] where k is output column, i is output row.
    # C[i][k] = sum_j A[i][j] B[j][k]
    # W index: k*n + i = k*3 + i
    
    # For the 2x2 sub-problem A_L * B00:
    # A_L is 2x2 (rows 0-1 of A, cols 0-1), B00 is 2x2 (rows 0-1 of B, cols 0-1)
    # Result R is 2x2: R[i][j] = sum A_L[i][l] B00[l][j]
    # This contributes to C[i][j] for i in 0-1, j in 0-1
    # W index for C[i][j]: j*3 + i
    
    # So R[0][0] -> W[0], R[0][1] -> W[3], R[1][0] -> W[1], R[1][1] -> W[4]
    
    # For Strassen, the 7 products give:
    # C[0][0] = M1+M4-M5+M7
    # C[0][1] = M3+M5
    # C[1][0] = M2+M4
    # C[1][1] = M1-M2+M3+M6
    
    # Each M is a scalar product. In the tensor decomposition:
    # M1 contributes to C[0][0] (+1), C[1][1] (+1)
    # M2 contributes to C[1][0] (+1), C[1][1] (-1)
    # M3 contributes to C[0][1] (+1), C[1][1] (+1)
    # M4 contributes to C[0][0] (+1), C[1][0] (+1)
    # M5 contributes to C[0][0] (-1), C[0][1] (+1)
    # M6 contributes to C[1][1] (+1)
    # M7 contributes to C[0][0] (+1)
    
    # So for each 2x2x2 Strassen block, I have 7 rank-1 terms, each contributing to multiple W entries.
    
    # Let me restructure. For each 2x2x2 Strassen sub-problem, I'll add 7 columns to U, V, W.
    
    rank = 44
    U = [[0]*rank for _ in range(12)]  # 3*4
    V = [[0]*rank for _ in range(16)]  # 4*4
    W = [[0]*rank for _ in range(12)]  # 4*3
    
    col = 0
    
    def add_strassen(a_idx, b_idx, w_map):
        """
        a_idx: tuple of 4 U indices for the 2x2 A block (row0col0, row0col1, row1col0, row1col1)
        b_idx: tuple of 4 V indices for the 2x2 B block
        w_map: dict mapping (row, col) of 2x2 result to W index
        """
        nonlocal col
        a00, a01, a10, a11 = a_idx
        b00, b01, b10, b11 = b_idx
        w00 = w_map[(0,0)]
        w01 = w_map[(0,1)]
        w10 = w_map[(1,0)]
        w11 = w_map[(1,1)]
        
        # M1 = (a00+a11)(b00+b11)
        U[a00][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        V[b11][col] += 1
        W[w00][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M2 = (a10+a11)*b00
        U[a10][col] += 1
        U[a11][col] += 1
        V[b00][col] += 1
        W[w10][col] += 1
        W[w11][col] += -1
        col += 1
        
        # M3 = a00*(b10-b11)
        U[a00][col] += 1
        V[b10][col] += 1
        V[b11][col] += -1
        W[w01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M4 = a11*(b01-b00)
        U[a11][col] += 1
        V[b01][col] += 1
        V[b00][col] += -1
        W[w00][col] += 1
        W[w10][col] += 1
        col += 1
        
        # M5 = (a00+a01)*b11
        U[a00][col] += 1
        U[a01][col] += 1
        V[b11][col] += 1
        W[w00][col] += -1
        W[w01][col] += 1
        col += 1
        
        # M6 = (a10-a00)*(b00+b01)
        U[a10][col] += 1
        U[a00][col] += -1
        V[b00][col] += 1
        V[b01][col] += 1
        W[w11][col] += 1
        col += 1
        
        # M7 = (a01-a11)*(b10+b11)
        U[a01][col] += 1
        U[a11][col] += -1
        V[b10][col] += 1
        V[b11][col] += 1
        W[w00][col] += 1
        col += 1
    
    # Part 1: 2x4x4 (rows 0,1 of A)
    # A_L: A[0][0]=U[0], A[0][1]=U[1], A[1][0]=U[4], A[1][1]=U[5]
    # A_R: A[0][2]=U[2], A[0][3]=U[3], A[1][2]=U[6], A[1][3]=U[7]
    # B00: B[0][0]=V[0], B[0][1]=V[1], B[1][0]=V[4], B[1][1]=V[5]
    # B01: B[0][2]=V[2], B[0][3]=V[3], B[1][2]=V[6], B[1][3]=V[7]
    # B10: B[2][0]=V[8], B[2][1]=V[9], B[3][0]=V[12], B[3][1]=V[13]
    # B11: B[2][2]=V[10], B[2][3]=V[11], B[3][2]=V[14], B[3][3]=V[15]
    
    # A_L * B00 -> C[0..1][0..1]: W[j*3+i]
    # (0,0)->W[0], (0,1)->W[3], (1,0)->W[1], (1,1)->W[4]
    add_strassen((0,1,4,5), (0,1,4,5), {(0,0):0, (0,1):3, (1,0):1, (1,1):4})
    
    # A_R * B10 -> C[0..1][0..1]: same W mapping
    add_strassen((2,3,6,7), (8,9,12,13), {(0,0):0, (0,1):3, (1,0):1, (1,1):4})
    
    # A_L * B01 -> C[0..1][2..3]: W[(j+2)*3+i]
    # (0,0)->W[6], (0,1)->W[9], (1,0)->W[7], (1,1)->W[10]
    add_strassen((0,1,4,5), (2,3,6,7), {(0,0):6, (0,1):9, (1,0):7, (1,1):10})
    
    # A_R * B11 -> C[0..1][2..3]: same W mapping
    add_strassen((2,3,6,7), (10,11,14,15), {(0,0):6, (0,1):9, (1,0):7, (1,1):10})
    
    # Part 2: 1x4x4 (row 2 of A)
    # A[2][j] = U[8+j] for j=0..3
    # B[j][k] = V[j*4+k]
    # C[2][k] = W[k*3+2]
    for j in range(4):
        for k in range(4):
            U[8+j][col] = 1
            V[j*4+k][col] = 1
            W[k*3+2][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
