"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partition B (2x10) into five 2x2 blocks along the column axis
    nblocks = p // m  # 5 blocks of size m x m = 2x2
    rank = 7 * nblocks  # 35

    U = [[0] * rank for _ in range(n * m)]   # shape (4, 35)
    V = [[0] * rank for _ in range(m * p)]   # shape (20, 35)
    W = [[0] * rank for _ in range(p * n)]   # shape (20, 35)

    # Strassen rank-7 scheme for 2x2 * 2x2
    # Inputs a00, a01, a10, a11 (from A, indexed by (i,j) -> row i*m+j)
    # Inputs b00, b01, b10, b11 (from B block, indexed by (j, col) -> row j*p + col)
    # Outputs c00, c01, c10, c11 (into C, indexed by (col, i) -> row col*n + i)
    #
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    #
    # c00 = M1 - M5 + M7
    # c01 = M3 + M5
    # c10 = M2 + M4
    # c11 = M1 - M2 + M6

    for block in range(nblocks):
        base_col = block * m  # starting column in B

        # For each Strassen step t in 0..6, assign column = block*7 + t
        col = block * 7 + 0
        # M1: U: a00=1, a11=1; V: b00=1, b11=1; W: c00=1, c11=1
        U[0][col] = 1    # a00
        U[3][col] = 1    # a11
        V[0 * p + base_col][col] = 1       # b00
        V[1 * p + base_col + 1][col] = 1   # b11
        W[base_col * n + 0][col] = 1       # c00
        W[(base_col + 1) * n + 1][col] = 1 # c11

        col = block * 7 + 1
        # M2: U: a10=1, a11=1; V: b00=1; W: c10=1, c11=-1
        U[2][col] = 1    # a10
        U[3][col] = 1    # a11
        V[0 * p + base_col][col] = 1       # b00
        W[base_col * n + 1][col] = 1       # c10
        W[(base_col + 1) * n + 1][col] = -1 # c11

        col = block * 7 + 2
        # M3: U: a00=1; V: b01=1, b11=-1; W: c01=1
        U[0][col] = 1    # a00
        V[0 * p + base_col + 1][col] = 1   # b01
        V[1 * p + base_col + 1][col] = -1  # b11
        W[(base_col + 1) * n + 0][col] = 1 # c01

        col = block * 7 + 3
        # M4: U: a11=1; V: b10=1, b00=-1; W: c10=1
        U[3][col] = 1    # a11
        V[1 * p + base_col][col] = 1       # b10
        V[0 * p + base_col][col] = -1      # b00
        W[base_col * n + 1][col] = 1       # c10

        col = block * 7 + 4
        # M5: U: a00=1, a01=1; V: b11=1; W: c00=-1, c01=1
        U[0][col] = 1    # a00
        U[1][col] = 1    # a01
        V[1 * p + base_col + 1][col] = 1   # b11
        W[base_col * n + 0][col] = -1      # c00
        W[(base_col + 1) * n + 0][col] = 1 # c01

        col = block * 7 + 5
        # M6: U: a10=1, a00=-1; V: b00=1, b01=1; W: c11=1
        U[2][col] = 1    # a10
        U[0][col] = -1   # a00
        V[0 * p + base_col][col] = 1       # b00
        V[0 * p + base_col + 1][col] = 1   # b01
        W[(base_col + 1) * n + 1][col] = 1 # c11

        col = block * 7 + 6
        # M7: U: a01=1, a11=-1; V: b10=1, b11=1; W: c00=1
        U[1][col] = 1    # a01
        U[3][col] = -1   # a11
        V[1 * p + base_col][col] = 1       # b10
        V[1 * p + base_col + 1][col] = 1   # b11
        W[base_col * n + 0][col] = 1       # c00

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
