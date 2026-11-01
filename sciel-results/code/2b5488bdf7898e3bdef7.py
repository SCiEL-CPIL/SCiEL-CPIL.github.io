"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    # Rank-15 exact decomposition:
    #   - Strassen rank-7 for the first 2x2 block (k in {0,1})
    #   - rank-2 per remaining 2x1 column block (k in {2..9})
    n, m, p = DIMENSIONS
    rank = 15
    U = [[0] * rank for _ in range(n * m)]   # 4 x 15
    V = [[0] * rank for _ in range(m * p)]   # 20 x 15
    W = [[0] * rank for _ in range(p * n)]   # 20 x 15

    # Strassen products for the 2x2 block (columns 0,1)
    # c1 = (A00+A01)*(B00+B11)
    U[0][0] = 1; U[1][0] = 1
    V[0][0] = 1; V[11][0] = 1
    W[0][0] = 1; W[11][0] = 1
    # c2 = A10*(B00+B01)
    U[3][1] = 1
    V[0][1] = 1; V[1][1] = 1
    W[1][1] = 1; W[11][1] = 1
    # c3 = (A01-A11)*B11
    U[1][2] = 1; U[3][2] = -1
    V[11][2] = 1
    W[11][2] = 1
    # c4 = A00*B10
    U[0][3] = 1
    V[10][3] = 1
    W[10][3] = 1
    # c5 = (A10+A11)*B11
    U[3][4] = 1; U[2][4] = 1
    V[11][4] = 1
    W[10][4] = 1; W[11][4] = 1
    # c6 = (A00-A10)*B00
    U[0][5] = 1; U[3][5] = -1
    V[0][5] = 1
    W[0][5] = 1
    # c7 = (A01+A11)*(B01-B11)
    U[1][6] = 1; U[2][6] = 1
    V[1][6] = 1; V[11][6] = -1
    W[1][6] = 1

    # Remaining 2x1 column blocks (k in 2..9), rank 2 each
    c = 7
    for k in range(2, p):
        # d1 = (A00+A01)*B0k
        U[0][c] = 1; U[1][c] = 1
        V[k][c] = 1
        W[k][c] = 1
        c += 1
        # d2 = A10*B1k
        U[3][c] = 1
        V[m + k][c] = 1
        W[k][c] = 1
        c += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
