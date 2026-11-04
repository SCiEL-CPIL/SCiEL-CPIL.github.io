"""Matrix multiplication 2x3 by 3x5 — rank-26 decomposition via Winograd 2x2x3 + Strassen 2x2x2."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 5
    rank = 26

    U = [[0] * rank for _ in range(n * m)]  # (6, 26)
    V = [[0] * rank for _ in range(m * p)]  # (15, 26)
    W = [[0] * rank for _ in range(p * n)]  # (10, 26)

    col = 0

    # ================================================================
    # Part 1: Winograd 2x2x3 decomposition
    # Computes C[:, 0:3] = A[:, 0:2] * B[0:2, 0:3]
    # A sub-matrix (2x2): A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[3], A[1,1]=U[4]
    # B sub-matrix (2x3): B[0,0]=V[0], B[0,1]=V[1], B[0,2]=V[2],
    #                     B[1,0]=V[5], B[1,1]=V[6], B[1,2]=V[7]
    # C outputs (2x3): C[0,0]=W[0], C[0,1]=W[2], C[0,2]=W[4],
    #                  C[1,0]=W[1], C[1,1]=W[3], C[1,2]=W[5]
    # ================================================================

    # M1 = (A11 + A22) * (B11 + B22)
    # A11=A[0,0], A22=A[1,1]; B11=B[0,0], B22=B[1,0]
    U[0][col] = 1;  U[4][col] = 1
    V[0][col] = 1;  V[5][col] = 1
    W[0][col] = 1;  W[2][col] = 1;  W[4][col] = 1
    W[1][col] = 1;  W[3][col] = 1;  W[5][col] = 1
    col += 1

    # M2 = A22 * (B21 - B11)
    # B21=B[1,1], B11=B[0,0]
    U[4][col] = 1
    V[6][col] = 1;  V[0][col] = -1
    W[1][col] = 1;  W[3][col] = 1;  W[5][col] = 1
    col += 1

    # M3 = (A11 - A21) * B22
    # A21=A[1,0]; B22=B[1,0]
    U[0][col] = 1;  U[3][col] = -1
    V[5][col] = 1
    W[0][col] = 1;  W[2][col] = 1;  W[4][col] = 1
    W[1][col] = 1;  W[3][col] = 1;  W[5][col] = 1
    col += 1

    # M4 = A11 * (B12 - B22)
    # B12=B[0,2], B22=B[1,0]
    U[0][col] = 1
    V[2][col] = 1;  V[5][col] = -1
    W[0][col] = 1;  W[2][col] = 1;  W[4][col] = 1
    col += 1

    # M5 = (A12 + A22) * (B11 + B12)
    # A12=A[0,1]; B11=B[0,0], B12=B[0,2]
    U[1][col] = 1;  U[4][col] = 1
    V[0][col] = 1;  V[2][col] = 1
    W[0][col] = 1;  W[2][col] = 1;  W[4][col] = 1
    W[1][col] = 1;  W[3][col] = 1;  W[5][col] = 1
    col += 1

    # M6 = (A11 - A21) * (B21 + B22)
    # B21=B[1,1], B22=B[1,0]
    U[0][col] = 1;  U[3][col] = -1
    V[6][col] = 1;  V[5][col] = 1
    W[1][col] = 1;  W[3][col] = 1;  W[5][col] = 1
    col += 1

    # M7 = (A12 - A22) * (B11 - B21)
    # B11=B[0,0], B21=B[1,1]
    U[1][col] = 1;  U[4][col] = -1
    V[0][col] = 1;  V[6][col] = -1
    W[0][col] = 1;  W[2][col] = 1;  W[4][col] = 1
    W[1][col] = 1;  W[3][col] = 1;  W[5][col] = 1
    col += 1

    # M8 = (A12 + A21 + A22) * B12
    # A12=A[0,1], A21=A[1,0], A22=A[1,1]; B12=B[0,2]
    U[1][col] = 1;  U[3][col] = 1;  U[4][col] = 1
    V[2][col] = 1
    W[0][col] = 1;  W[2][col] = 1;  W[4][col] = 1
    col += 1

    # M9 = (A12 + A21) * B21
    # A12=A[0,1], A21=A[1,0]; B21=B[1,1]
    U[1][col] = 1;  U[3][col] = 1
    V[6][col] = 1
    W[1][col] = 1;  W[3][col] = 1;  W[5][col] = 1
    col += 1

    # M10 = A12 * B11
    # A12=A[0,1]; B11=B[0,0]
    U[1][col] = 1
    V[0][col] = 1
    W[0][col] = 1;  W[2][col] = 1;  W[4][col] = 1
    col += 1

    # ================================================================
    # Part 2: Strassen 2x2x2 decomposition
    # Computes C[:, 3:5] = A[:, 0:2] * B[0:2, 3:5]
    # A sub-matrix (2x2): same as above
    # B sub-matrix (2x2): B[0,3]=V[3], B[0,4]=V[4], B[1,3]=V[8], B[1,4]=V[9]
    # C outputs (2x2): C[0,3]=W[6], C[0,4]=W[8], C[1,3]=W[7], C[1,4]=W[9]
    # ================================================================

    # M1 = (A11 + A22) * (B11 + B22)
    # B11=B[0,3], B22=B[1,3]
    U[0][col] = 1;  U[4][col] = 1
    V[3][col] = 1;  V[8][col] = 1
    W[6][col] = 1;  W[8][col] = 1
    W[7][col] = 1;  W[9][col] = 1
    col += 1

    # M2 = A22 * (B21 - B11)
    # B21=B[1,4], B11=B[0,3]
    U[4][col] = 1
    V[9][col] = 1;  V[3][col] = -1
    W[7][col] = 1;  W[9][col] = 1
    col += 1

    # M3 = (A11 - A21) * B22
    # A21=A[1,0]; B22=B[1,3]
    U[0][col] = 1;  U[3][col] = -1
    V[8][col] = 1
    W[6][col] = 1;  W[8][col] = 1
    W[7][col] = 1;  W[9][col] = 1
    col += 1

    # M4 = A11 * (B12 - B22)
    # B12=B[0,4], B22=B[1,3]
    U[0][col] = 1
    V[4][col] = 1;  V[8][col] = -1
    W[6][col] = 1;  W[8][col] = 1
    col += 1

    # M5 = (A12 + A22) * (B11 + B12)
    # A12=A[0,1]; B11=B[0,3], B12=B[0,4]
    U[1][col] = 1;  U[4][col] = 1
    V[3][col] = 1;  V[4][col] = 1
    W[6][col] = 1;  W[8][col] = 1
    W[7][col] = 1;  W[9][col] = 1
    col += 1

    # M6 = (A11 - A21) * (B21 + B22)
    # B21=B[1,4], B22=B[1,3]
    U[0][col] = 1;  U[3][col] = -1
    V[9][col] = 1;  V[8][col] = 1
    W[7][col] = 1;  W[9][col] = 1
    col += 1

    # M7 = (A12 - A22) * (B11 - B21)
    # B11=B[0,3], B21=B[1,4]
    U[1][col] = 1;  U[4][col] = -1
    V[3][col] = 1;  V[9][col] = -1
    W[6][col] = 1;  W[8][col] = 1
    W[7][col] = 1;  W[9][col] = 1
    col += 1

    # ================================================================
    # Part 3: Schoolbook for A[:, 2] * B[2, :] -> C[:, 0:5]
    # A[0,2]=U[2], A[1,2]=U[5]
    # B[2,0]=V[10], B[2,1]=V[11], B[2,2]=V[12], B[2,3]=V[13], B[2,4]=V[14]
    # C outputs: C[0,k]=W[2k], C[1,k]=W[2k+1]
    # ================================================================
    for k in range(p):
        # C[0,k] += A[0,2] * B[2,k]
        U[2][col] = 1
        V[10 + k][col] = 1
        W[2 * k][col] = 1
        col += 1

        # C[1,k] += A[1,2] * B[2,k]
        U[5][col] = 1
        V[10 + k][col] = 1
        W[2 * k + 1][col] = 1
        col += 1

    assert col == rank, f"Column count {col} != rank {rank}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
