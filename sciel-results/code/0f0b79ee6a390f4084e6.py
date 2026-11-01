"""Matrix multiplication 2,2,4 with rank-14 decomposition."""
import time

DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=4
    rank = 14

    # U: (n*m, rank) = (4, 14)
    # V: (m*p, rank) = (8, 14)
    # W: (p*n, rank) = (8, 14)

    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]

    # Standard Strassen for 2x2 matrices uses 7 multiplications.
    # For 2x2 * 2x4, we can view this as two independent 2x2 multiplications
    # (for columns 0-1 and columns 2-3) plus potential cross-term savings.
    #
    # A simpler approach: use 7 Strassen terms for the first 2x2 block (cols 0,1)
    # and 7 for the second 2x2 block (cols 2,3) = 14 total.
    #
    # Let's use the standard Strassen construction for each block.
    #
    # For a 2x2 matrix product C = A*B where:
    # A = [[a00, a01], [a10, a11]]  (row i, col j)
    # B = [[b00, b01], [b10, b11]]  (row j, col k)
    #
    # Strassen's 7 products:
    # M1 = (a00 + a11) * (b00 + b11)
    # M2 = (a10 + a11) * b00
    # M3 = a00 * (b01 - b11)
    # M4 = a11 * (b10 - b00)
    # M5 = (a00 + a01) * b11
    # M6 = (a10 - a00) * (b00 + b01)
    # M7 = (a01 - a11) * (b10 + b11)
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # U indices: i*m+j where i is row of A, j is col of A
    #   U[0] = row 0, col 0 (a00)
    #   U[1] = row 0, col 1 (a01)
    #   U[2] = row 1, col 0 (a10)
    #   U[3] = row 1, col 1 (a11)
    #
    # V indices: j*p+k where j is row of B, k is col of B
    #   V[0] = row 0, col 0 (b00)
    #   V[1] = row 0, col 1 (b01)
    #   V[2] = row 0, col 2 (b02)
    #   V[3] = row 0, col 3 (b03)
    #   V[4] = row 1, col 0 (b10)
    #   V[5] = row 1, col 1 (b11)
    #   V[6] = row 1, col 2 (b12)
    #   V[7] = row 1, col 3 (b13)
    #
    # W indices: k*n+i where k is col of C, i is row of C
    #   W[0] = col 0, row 0 (C[0,0])
    #   W[1] = col 0, row 1 (C[1,0])
    #   W[2] = col 1, row 0 (C[0,1])
    #   W[3] = col 1, row 1 (C[1,1])
    #   W[4] = col 2, row 0 (C[0,2])
    #   W[5] = col 2, row 1 (C[1,2])
    #   W[6] = col 3, row 0 (C[0,3])
    #   W[7] = col 3, row 1 (C[1,3])

    # Block 1: columns 0 and 1 of B
    # M1 = (a00 + a11) * (b00 + b11)
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[5][0] = 1
    W[0][0] = 1; W[3][0] = 1  # C[0,0] and C[1,1]

    # M2 = (a10 + a11) * b00
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[1][1] = 1; W[3][1] = -1  # C[1,0] and C[1,1]

    # M3 = a00 * (b01 - b11)
    U[0][2] = 1
    V[1][2] = 1; V[5][2] = -1
    W[2][2] = 1; W[3][2] = 1  # C[0,1] and C[1,1]

    # M4 = a11 * (b10 - b00)
    U[3][3] = 1
    V[4][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[1][3] = 1  # C[0,0] and C[1,0]

    # M5 = (a00 + a01) * b11
    U[0][4] = 1; U[1][4] = 1
    V[5][4] = 1
    W[0][4] = -1; W[2][4] = 1  # C[0,0] and C[0,1]

    # M6 = (a10 - a00) * (b00 + b01)
    U[2][5] = 1; U[0][5] = -1
    V[0][5] = 1; V[1][5] = 1
    W[3][5] = 1  # C[1,1]

    # M7 = (a01 - a11) * (b10 + b11)
    U[1][6] = 1; U[3][6] = -1
    V[4][6] = 1; V[5][6] = 1
    W[0][6] = 1  # C[0,0]

    # Block 2: columns 2 and 3 of B
    # M8 = (a00 + a11) * (b02 + b13)
    U[0][7] = 1; U[3][7] = 1
    V[2][7] = 1; V[7][7] = 1
    W[4][7] = 1; W[7][7] = 1  # C[0,2] and C[1,3]

    # M9 = (a10 + a11) * b02
    U[2][8] = 1; U[3][8] = 1
    V[2][8] = 1
    W[5][8] = 1; W[7][8] = -1  # C[1,2] and C[1,3]

    # M10 = a00 * (b03 - b13)
    U[0][9] = 1
    V[3][9] = 1; V[7][9] = -1
    W[6][9] = 1; W[7][9] = 1  # C[0,3] and C[1,3]

    # M11 = a11 * (b12 - b02)
    U[3][10] = 1
    V[6][10] = 1; V[2][10] = -1
    W[4][10] = 1; W[5][10] = 1  # C[0,2] and C[1,2]

    # M12 = (a00 + a01) * b13
    U[0][11] = 1; U[1][11] = 1
    V[7][11] = 1
    W[4][11] = -1; W[6][11] = 1  # C[0,2] and C[0,3]

    # M13 = (a10 - a00) * (b02 + b03)
    U[2][12] = 1; U[0][12] = -1
    V[2][12] = 1; V[3][12] = 1
    W[7][12] = 1  # C[1,3]

    # M14 = (a01 - a11) * (b12 + b13)
    U[1][13] = 1; U[3][13] = -1
    V[6][13] = 1; V[7][13] = 1
    W[4][13] = 1  # C[0,2]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
