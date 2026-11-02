"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    # Rank-44 decomposition:
    #   Split the 2x3x8 multiplication into:
    #     (a) A1 (2x1) * B1 (1x8): schoolbook, rank 16
    #     (b) A2 (2x2) * B2 (2x8): four 2x2x2 Strassen blocks, rank 28
    #   Total rank = 16 + 28 = 44
    n, m, p = 2, 3, 8
    rank = 44
    U = [[0] * rank for _ in range(n * m)]  # 6 x 44
    V = [[0] * rank for _ in range(m * p)]  # 24 x 44
    W = [[0] * rank for _ in range(p * n)]  # 16 x 44

    # Part (a): A1 (rows 0,1 of A, i.e. i=0,1 with j=0) * B1 (row 0 of B, i.e. j=0, all k)
    # U index: i*m + j = i*3 + 0 = 3*i  for i in {0,1}
    # V index: j*p + k = 0*8 + k = k     for k in 0..7
    # W index: k*n + i = 2*k + i         for k in 0..7, i in {0,1}
    col = 0
    for i in range(n):       # 2
        for k in range(p):   # 8
            U[3 * i][col] = 1
            V[k][col] = 1
            W[2 * k + i][col] = 1
            col += 1
    # col is now 16

    # Part (b): A2 (rows 0,1 of A with j=1,2, i.e. i*3+1, i*3+2) * B2 (rows 1,2 of B, i.e. j=1,2, all k)
    # Split B2 into four 2x2 blocks by columns: block b covers columns 2b, 2b+1
    # For each block b (0..3), apply Strassen to compute A2 * B2_b (2x2 result)
    # Strassen for 2x2:
    #   M1 = (A00+A11)(B00+B11)
    #   M2 = (A10+A11)B00
    #   M3 = A00(B10-B11)
    #   M4 = A11(B01-B10)
    #   M5 = (A00+A01)B11
    #   M6 = (A10-A00)(B00+B10)
    #   M7 = (A01-A11)(B01+B11)
    #   C00 = M1 + M4 - M5 + M7
    #   C01 = M3 + M5
    #   C10 = M2 + M4
    #   C11 = M1 - M2 + M3 + M6
    #
    # A2 indices in U: row i*3 + j where j in {1,2}
    #   A00 = U[1], A01 = U[2], A10 = U[4], A11 = U[5]
    # B2_b indices in V: row j*8 + k where j in {1,2}, k in {2b, 2b+1}
    #   B00 = V[8+2b], B01 = V[8+2b+1], B10 = V[16+2b], B11 = V[16+2b+1]
    # C block b indices in W: row k*n + i = 2k + i, k in {2b, 2b+1}, i in {0,1}
    #   C00 = W[2*(2b)+0] = W[4b]
    #   C01 = W[2*(2b+1)+0] = W[4b+2]
    #   C10 = W[2*(2b)+1] = W[4b+1]
    #   C11 = W[2*(2b+1)+1] = W[4b+3]

    for b in range(4):
        # Strassen factors for block b
        # M1 = (A00+A11)(B00+B11)
        col += 0  # will increment below
        U[1][col] += 1;  U[5][col] += 1
        V[8 + 2*b][col] += 1;  V[16 + 2*b + 1][col] += 1
        W[4*b][col] += 1;  W[4*b+2][col] += 1;  W[4*b+1][col] += 1;  W[4*b+3][col] += 1
        col += 1

        # M2 = (A10+A11)B00
        U[4][col] += 1;  U[5][col] += 1
        V[8 + 2*b][col] += 1
        W[4*b+1][col] += 1;  W[4*b+3][col] -= 1
        col += 1

        # M3 = A00(B10-B11)
        U[1][col] += 1
        V[16 + 2*b][col] += 1;  V[16 + 2*b + 1][col] -= 1
        W[4*b+2][col] += 1;  W[4*b+3][col] += 1
        col += 1

        # M4 = A11(B01-B10)
        U[5][col] += 1
        V[8 + 2*b + 1][col] += 1;  V[16 + 2*b][col] -= 1
        W[4*b][col] += 1;  W[4*b+1][col] += 1
        col += 1

        # M5 = (A00+A01)B11
        U[1][col] += 1;  U[2][col] += 1
        V[16 + 2*b + 1][col] += 1
        W[4*b][col] -= 1;  W[4*b+2][col] += 1
        col += 1

        # M6 = (A10-A00)(B00+B10)
        U[4][col] += 1;  U[1][col] -= 1
        V[8 + 2*b][col] += 1;  V[16 + 2*b][col] += 1
        W[4*b+3][col] += 1
        col += 1

        # M7 = (A01-A11)(B01+B11)
        U[2][col] += 1;  U[5][col] -= 1
        V[8 + 2*b + 1][col] += 1;  V[16 + 2*b + 1][col] += 1
        W[4*b][col] += 1;  W[4*b+2][col] += 1
        col += 1

    # col should be 16 + 28 = 44
    assert col == rank, f"Expected rank {rank}, got {col}"
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
