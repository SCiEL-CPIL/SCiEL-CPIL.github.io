"""Four Russians / block-Strassen rank-18 decomposition for M_{2,3,7}."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    assert (n, m, p) == (2, 3, 7)

    rank = 18
    U = [[0] * rank for _ in range(n * m)]   # 6 rows x 18
    V = [[0] * rank for _ in range(m * p)]   # 21 rows x 18
    W = [[0] * rank for _ in range(p * n)]   # 14 rows x 18

    def add_strip_block(i, b0, cols, col):
        # Block 'cols' (list of k's), for row-strip i of A.
        # Returns the next free column.
        s1 = [1, 1, 1]
        s2 = [1, 0, 1]
        for j in range(3):
            U[i * m + j][col]     = s1[j]
            U[i * m + j][col + 1] = s2[j]
        s3 = [0, 0, 0]
        s4 = [0, 0, 0]
        for k in cols:
            s3[k] = 1
            s4[k] = 1
        for k in range(p):
            V[b0 * p + k][col]     = s3[k]
            V[b0 * p + k][col + 1] = s4[k]
        # m1 = s1*s3 covers all cols, m2 = s2*s4 covers first+last.
        for k in cols:
            W[k * n + i][col]     = 1
            W[k * n + i][col + 1] = (1 if (k == cols[0] or k == cols[-1]) else 0)
        return col + 2

    col = 0
    # Block 1: 4 columns (k=0,1,2,3)
    for i in range(2):
        col = add_strip_block(i, 0, [0, 1, 2, 3], col)
    # Block 2: 3 columns (k=4,5,6)
    for i in range(2):
        col = add_strip_block(i, 4, [4, 5, 6], col)
    # Block 3: 1 column (k=6) -> single scalar product each strip
    for i in range(2):
        U[i * m][col] = 1
        V[6 * p + 6][col] = 1
        W[6 * n + i][col] = 1
        col += 1

    assert col == rank, (col, rank)
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
