"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    # Strassen-based rank-22 decomposition for 3x4x4 matrix multiplication.
    # Split into 4 blocks: (rows 0,1 x cols 0,1) Strassen rank 7,
    # (rows 0,1 x cols 2,3) Strassen rank 7,
    # (row 2 x cols 0,1) schoolbook rank 4,
    # (row 2 x cols 2,3) schoolbook rank 4.
    n, m, p = 3, 4, 4
    rank = 22
    U = [[0] * rank for _ in range(n * m)]  # 12 x 22
    V = [[0] * rank for _ in range(m * p)]  # 16 x 22
    W = [[0] * rank for _ in range(p * n)]  # 12 x 22

    # Strassen 2x2x2 coefficients (u, v, w for each of 7 terms)
    # M1 = (a11+a22)(b11+b22), M2 = (a21+a22)b11, M3 = a11(b12-b22),
    # M4 = a22(b21-b11), M5 = (a11+a12)b22, M6 = (a21-a11)(b11+b12),
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7, C12 = M3+M5, C21 = M2+M4, C22 = M1-M2+M3+M6
    strassen_U = [
        [1, 0, 1, 1, 1, 0, 1],   # a11
        [0, 1, 0, 0, 0, 1, 1],   # a12
        [0, 0, 0, 0, 0, 0, 0],   # a21 (not used in this mapping)
        [1, 1, 1, 1, 1, 1, 1],   # a22
    ]
    # Wait, let me redo. Standard Strassen:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    # U coefficients for a11, a12, a21, a22 in each Mi
    U_11 = [1, 0, 1, 0, 1, -1, 0]
    U_12 = [0, 0, 0, 0, 1, 0, 1]
    U_21 = [0, 1, 0, 0, 0, 1, 0]
    U_22 = [1, 1, 0, 1, 0, 0, -1]

    # V coefficients for b11, b12, b21, b22 in each Mi
    V_11 = [1, 1, 0, -1, 0, 1, 0]
    V_12 = [0, 0, 1, 0, 0, 1, 0]
    V_21 = [0, 0, 0, 1, 0, 0, 1]
    V_22 = [1, -1, -1, 0, 1, 0, 0]

    # W coefficients for C11, C12, C21, C22 in each Mi
    W_11 = [1, 0, 0, 1, -1, 0, 1]
    W_12 = [0, 0, 1, 0, 1, 0, 0]
    W_21 = [0, 1, 0, 1, 0, 0, 0]
    W_22 = [1, -1, 1, 0, 0, 1, 0]

    def add_strassen_block(U_off_a, V_off_b, W_off_c):
        """Add a Strassen 2x2x2 block.
        U_off_a: offset in U for a11 (U row for A[row0][col0])
        V_off_b: offset in V for b11 (V row for B[row0][col0])
        W_off_c: offset in W for C11 (W row for C[col0][row0])
        """
        for r in range(7):
            col = 0  # We'll track column separately
        # Actually let me just add directly with a column counter approach
        return  # placeholder

    # Let me just build it explicitly with column tracking
    col = 0

    # Block 1: rows {0,1} x cols {0,1}, Strassen
    # U rows: A[0,0]=0, A[0,1]=1, A[1,0]=4, A[1,1]=5
    # V rows: B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # W rows: C[0,0]=0, C[0,1]=1, C[1,0]=4, C[1,1]=5
    for r in range(7):
        U[0][col] += U_11[r]
        U[1][col] += U_12[r]
        U[4][col] += U_21[r]
        U[5][col] += U_22[r]
        V[0][col] += V_11[r]
        V[1][col] += V_12[r]
        V[4][col] += V_21[r]
        V[5][col] += V_22[r]
        W[0][col] += W_11[r]
        W[1][col] += W_12[r]
        W[4][col] += W_21[r]
        W[5][col] += W_22[r]
        col += 1

    # Block 2: rows {0,1} x cols {2,3}, Strassen
    # U rows: A[0,2]=2, A[0,3]=3, A[1,2]=6, A[1,3]=7
    # V rows: B[2,0]=8, B[2,1]=9, B[3,0]=12, B[3,1]=13
    # W rows: C[0,2]=2, C[0,3]=3, C[1,2]=6, C[1,3]=7
    for r in range(7):
        U[2][col] += U_11[r]
        U[3][col] += U_12[r]
        U[6][col] += U_21[r]
        U[7][col] += U_22[r]
        V[8][col] += V_11[r]
        V[9][col] += V_12[r]
        V[12][col] += V_21[r]
        V[13][col] += V_22[r]
        W[2][col] += W_11[r]
        W[3][col] += W_12[r]
        W[6][col] += W_21[r]
        W[7][col] += W_22[r]
        col += 1

    # Block 3: row {2} x cols {0,1}, schoolbook rank 4
    # U rows: A[2,0]=8, A[2,1]=9
    # V rows: B[0,0]=0, B[0,1]=1, B[1,0]=4, B[1,1]=5
    # W rows: C[2,0]=8, C[2,1]=9
    for j in range(2):
        for k in range(2):
            U[8 + j][col] = 1
            V[k * 2 + j][col] = 1  # B[k][j] where k is row of B (0,1), j is col (0,1)
            W[8 + j][col] = 1  # C[j][2] where j is col of C (0,1)
            col += 1

    # Block 4: row {2} x cols {2,3}, schoolbook rank 4
    # U rows: A[2,2]=10, A[2,3]=11
    # V rows: B[2,0]=8, B[2,1]=9, B[3,0]=12, B[3,1]=13
    # W rows: C[2,2]=10, C[2,3]=11
    for j in range(2):
        for k in range(2):
            U[10 + j][col] = 1
            V[(2 + k) * 2 + j][col] = 1  # B[2+k][j] where k is 0,1 -> rows 2,3; j is col 0,1
            W[10 + j][col] = 1  # C[2+j][2] where j is col 0,1 -> cols 2,3
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
