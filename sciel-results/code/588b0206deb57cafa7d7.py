"""Block-Strassen decomposition for 4x4 x 4x6 matrix multiplication."""
DIMENSIONS = [4, 4, 6]

# Strassen's rank-7 decomposition for 2x2 x 2x2 matrix multiplication.
# For multiplying A (2x2) by B (2x2) to get C (2x2):
#   C[0,0] = M1 + M4 - M5 - M7
#   C[0,1] = M3 + M5
#   C[1,0] = M2 + M4
#   C[1,1] = M1 - M3 + M6 + M7
# where:
#   M1 = (A00+A11)*(B00+B11)
#   M2 = (A10+A11)*B00
#   M3 = A00*(B01-B11)
#   M4 = A11*(B10-B00)
#   M5 = (A00+A01)*B11
#   M6 = (A10-A00)*(B00+B01)
#   M7 = (A01-A11)*(B10+B11)

# U factors: (row, col) -> coefficient for each of the 7 M's
# Row index = i*2 + j (i is row of A, j is col of A), col = M index
STRASSEN_U = [
    # A00: row 0
    [1, 0, 1, 0, 0, -1, 0],
    # A01: row 1
    [0, 0, 0, 0, 1, 0, 1],
    # A10: row 2
    [0, 1, 0, 0, 0, 1, 0],
    # A11: row 3
    [1, 1, 0, 1, 0, 0, -1],
]

# V factors: (row, col) -> coefficient
# Row index = j*2 + k (j is row of B, k is col of B), col = M index
STRASSEN_V = [
    # B00: row 0
    [1, 1, 0, -1, 0, 1, 0],
    # B01: row 1
    [0, 0, 1, 0, 0, 1, 0],
    # B10: row 2
    [-1, 0, 0, 1, 0, 0, 1],
    # B11: row 3
    [1, 0, -1, 0, 1, 0, 1],
]

# W factors: (row, col) -> coefficient
# Row index = k*2 + i (k is row of C, i is col of C), col = M index
STRASSEN_W = [
    # C00: row 0
    [1, 0, 0, 1, -1, 0, -1],
    # C01: row 1
    [0, 0, 1, 0, 1, 0, 0],
    # C10: row 2
    [0, 1, 0, 1, 0, 0, 0],
    # C11: row 3
    [1, 0, -1, 0, 0, 1, 1],
]


def construct():
    n, m, p = DIMENSIONS
    # Block sizes
    bn, bm, bp = 2, 2, 2
    # Number of blocks
    nb = n // bn  # 2
    mb = m // bm  # 2
    pb = p // bp  # 3
    total_blocks = nb * mb * pb  # 12
    rank = total_blocks * 7  # 84

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for bi in range(nb):
        for bj in range(mb):
            for bk in range(pb):
                # For each of the 7 Strassen multiplications in this block
                for s in range(7):
                    # U: row = (bi*bn + i) * m + (bj*bm + j)
                    for i in range(bn):
                        for j in range(bm):
                            row_u = (bi * bn + i) * m + (bj * bm + j)
                            U[row_u][col] = STRASSEN_U[i * 2 + j][s]
                    # V: row = (bj*bm + j) * p + (bk*bp + k)
                    for j in range(bm):
                        for k in range(bp):
                            row_v = (bj * bm + j) * p + (bk * bp + k)
                            V[row_v][col] = STRASSEN_V[j * 2 + k][s]
                    # W: row = (bk*bp + k) * n + (bi*bn + i)
                    for k in range(bp):
                        for i in range(bn):
                            row_w = (bk * bp + k) * n + (bi * bn + i)
                            W[row_w][col] = STRASSEN_W[k * 2 + i][s]
                    col += 1

    return {"U": U, "V": V, "W": W}
