"""Program for 3x4x7 matrix multiplication tensor decomposition."""

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START

def construct():
    n, m, p = 3, 4, 7
    # Target rank: 20
    # Block 1 (cols 0-2 of B): Rank 8 (7 Laderman + 1 Cross)
    # Block 2 (cols 3-5 of B): Rank 8 (7 Laderman + 1 Cross)
    # Block 3 (col 6 of B): Rank 4 (3 Dot + 1 Cross)
    total_rank = 20
    
    U = [[0.0] * total_rank for _ in range(n * m)]
    V = [[0.0] * total_rank for _ in range(m * p)]
    W = [[0.0] * total_rank for _ in range(p * n)]
    
    # Laderman's algorithm for 3x3x3
    # U_L (9x7), V_L (9x7), W_L (9x7)
    # Indices for U_L: i*3 + j (i,j in 0..2)
    # Indices for V_L: j*3 + k (j,k in 0..2)
    # Indices for W_L: k*3 + i (k,i in 0..2)
    
    # Laderman terms (standard formulation)
    # 1: (A00+A22)(B00+B22) -> C00, C11, C22, C02, C20
    # 2: (A01+A21)(B10+B12) -> C01, C12, C21, C00, C22
    # 3: (A10+A20)(B01+B02) -> C10, C21, C01, C11, C22
    # 4: (A00+A11)(B11+B20) -> C01, C10, C22, C02, C20
    # 5: (A11+A22)(B00+B21) -> C02, C21, C10, C11, C22
    # 6: (A01+A10)(B02+B12) -> C02, C20, C11, C01, C21
    # 7: (A12+A21)(B01+B20) -> C12, C21, C02, C10, C22
    
    # We will define the Laderman matrices explicitly.
    # U_L rows: 0..8 (i*3+j)
    # V_L rows: 0..8 (j*3+k)
    # W_L rows: 0..8 (k*3+i)
    
    U_L = [[0.0]*7 for _ in range(9)]
    V_L = [[0.0]*7 for _ in range(9)]
    W_L = [[0.0]*7 for _ in range(9)]
    
    # Term 0
    # A00, A22 -> U[0], U[8] = 1
    # B00, B22 -> V[0], V[8] = 1
    # C00, C11, C22, C02, C20 -> W[0], W[4], W[8], W[2], W[6] = 1
    U_L[0][0] = 1; U_L[8][0] = 1
    V_L[0][0] = 1; V_L[8][0] = 1
    W_L[0][0] = 1; W_L[4][0] = 1; W_L[8][0] = 1; W_L[2][0] = 1; W_L[6][0] = 1
    
    # Term 1
    # A01, A21 -> U[1], U[7] = 1
    # B10, B12 -> V[3], V[5] = 1
    # C01, C12, C21, C00, C22 -> W[1], W[5], W[7], W[0], W[8] = 1
    U_L[1][1] = 1; U_L[7][1] = 1
    V_L[3][1] = 1; V_L[5][1] = 1
    W_L[1][1] = 1; W_L[5][1] = 1; W_L[7][1] = 1; W_L[0][1] = 1; W_L[8][1] = 1
    
    # Term 2
    # A10, A20 -> U[3], U[6] = 1
    # B01, B02 -> V[1], V[2] = 1
    # C10, C21, C01, C11, C22 -> W[3], W[7], W[1], W[4], W[8] = 1
    U_L[3][2] = 1; U_L[6][2] = 1
    V_L[1][2] = 1; V_L[2][2] = 1
    W_L[3][2] = 1; W_L[7][2] = 1; W_L[1][2] = 1; W_L[4][2] = 1; W_L[8][2] = 1
    
    # Term 3
    # A00, A11 -> U[0], U[4] = 1
    # B11, B20 -> V[4], V[6] = 1
    # C01, C10, C22, C02, C20 -> W[1], W[3], W[8], W[2], W[6] = 1
    U_L[0][3] = 1; U_L[4][3] = 1
    V_L[4][3] = 1; V_L[6][3] = 1
    W_L[1][3] = 1; W_L[3][3] = 1; W_L[8][3] = 1; W_L[2][3] = 1; W_L[6][3] = 1
    
    # Term 4
    # A11, A22 -> U[4], U[8] = 1
    # B00, B21 -> V[0], V[7] = 1
    # C02, C21, C10, C11, C22 -> W[2], W[7], W[3], W[4], W[8] = 1
    U_L[4][4] = 1; U_L[8][4] = 1
    V_L[0][4] = 1; V_L[7][4] = 1
    W_L[2][4] = 1; W_L[7][4] = 1; W_L[3][4] = 1; W_L[4][4] = 1; W_L[8][4] = 1
    
    # Term 5
    # A01, A10 -> U[1], U[3] = 1
    # B02, B12 -> V[2], V[5] = 1
    # C02, C20, C11, C01, C21 -> W[2], W[6], W[4], W[1], W[7] = 1
    U_L[1][5] = 1; U_L[3][5] = 1
    V_L[2][5] = 1; V_L[5][5] = 1
    W_L[2][5] = 1; W_L[6][5] = 1; W_L[4][5] = 1; W_L[1][5] = 1; W_L[7][5] = 1
    
    # Term 6
    # A12, A21 -> U[5], U[7] = 1
    # B01, B20 -> V[1], V[6] = 1
    # C12, C21, C02, C10, C22 -> W[5], W[7], W[2], W[3], W[8] = 1
    U_L[5][6] = 1; U_L[7][6] = 1
    V_L[1][6] = 1; V_L[6][6] = 1
    W_L[5][6] = 1; W_L[7][6] = 1; W_L[2][6] = 1; W_L[3][6] = 1; W_L[8][6] = 1
    
    # Helper to add a block to U, V, W
    # U_block: 9xR, V_block: 9xR, W_block: 9xR
    # u_offset: row offset in U (0..3 for A rows)
    # v_offset: row offset in V (0..3 for B rows)
    # w_offset: row offset in W (0..6 for C rows)
    # col_offset: column offset in U,V,W
    
    def add_laderman(u_off, v_off, w_off, col_off):
        for r in range(9):
            for l in range(7):
                U[u_off + r // 3 * m + r % 3][col_off + l] += U_L[r][l]
                # V index: j*3+k. r in V_L is j*3+k.
                # V row index in full V: j*m + k.
                # Here j, k are 0..2.
                V[v_off + (r // 3) * m + (r % 3)][col_off + l] += V_L[r][l]
                # W index: k*3+i. r in W_L is k*3+i.
                # W row index in full W: k*n + i.
                # Here k, i are 0..2.
                W[w_off + (r // n) * n + (r % n)][col_off + l] += W_L[r][l]

    def add_cross_term(u_off, v_off, w_off, col_off):
        # Term: A[u_off+i, 3] * B[3, v_off+k] -> C[w_off+k, i]
        # U index: i*m + 3. i in 0..2.
        # V index: 3*m + (v_off+k). k in 0..2.
        # W index: (w_off+k)*n + i.
        for i in range(n):
            U[u_off + i * m + 3][col_off] += 1.0
        for k in range(n):
            V[3 * m + v_off + k][col_off] += 1.0
        for i in range(n):
            for k in range(n):
                W[(w_off + k) * n + i][col_off] += 1.0

    def add_dot_terms(u_off, v_off, w_off, col_off):
        # Terms for 3x3x1: C_i = sum A[i,k] B[k, v_off]
        # Rank 3. One term per k.
        for k in range(n):
            U[u_off + k * m + k][col_off + k] += 1.0 # Wait, U index is i*m+j. Here j=k?
            # A is 3x3 (first 3 cols). Index i*m + k.
            # We need U[i*m+k, l] = 1 if l corresponds to this k.
            # Let's use separate columns for each k.
            pass
            
        # Corrected Dot Terms Logic:
        # C_i = sum_{k=0}^2 A[i,k] B[k, v_off]
        # Term l=k:
        # U[i*m+k, col_off+k] = 1 for all i? No.
        # U[i*m+k, l] should be 1 if the term involves A[i,k].
        # For term k, U[i*m+k, col_off+k] = 1.
        # V[k*m + v_off, col_off+k] = 1.
        # W[(v_off)*n + i, col_off+k] = 1.
        
        for k in range(n):
            for i in range(n):
                U[u_off + i * m + k][col_off + k] += 1.0
            V[k * m + v_off][col_off + k] += 1.0
            for i in range(n):
                W[v_off * n + i][col_off + k] += 1.0

    def add_cross_term_1(u_off, v_off, w_off, col_off):
        # Term for 3x1x1: C_i = A[i,3] B[3, v_off]
        # Rank 1.
        for i in range(n):
            U[u_off + i * m + 3][col_off] += 1.0
        V[3 * m + v_off][col_off] += 1.0
        for i in range(n):
            W[v_off * n + i][col_off] += 1.0

    # Block 1: B cols 0-2
    add_laderman(0, 0, 0, 0)
    add_cross_term(0, 0, 0, 7)
    
    # Block 2: B cols 3-5
    add_laderman(0, 3, 3, 8)
    add_cross_term(0, 3, 3, 15)
    
    # Block 3: B col 6
    add_dot_terms(0, 6, 6, 16)
    add_cross_term_1(0, 6, 6, 19)

    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
