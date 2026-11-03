"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-58 decomposition using two Strassen 2x2x2 blocks.
    Block 1: i in {0,1}, j in {0,1}, k in {0,1}
    Block 2: i in {0,1}, j in {2,3}, k in {2,3}
    Remaining: schoolbook
    """
    n, m, p = DIMENSIONS  # 3, 4, 5
    rank = 58
    U = [[0] * rank for _ in range(n * m)]  # 12 x 58
    V = [[0] * rank for _ in range(m * p)]  # 20 x 58
    W = [[0] * rank for _ in range(p * n)]  # 15 x 58
    col = 0

    def strassen_block(i_off, j_off, k_off):
        nonlocal col
        # 7 Strassen multipliers for 2x2x2
        # Using standard decomposition:
        # M0 = (a00+a11)(b00+b11)
        # M1 = (a10+a11)b00
        # M2 = a00(b10-b11)
        # M3 = a11(b01-b10)
        # M4 = (a00+a01)b11
        # M5 = (a10-a00)(b00+b10)
        # M6 = (a01-a11)(b01+b11)
        # c00 = M0 + M3 - M4 - M6
        # c01 = M2 + M4
        # c10 = M1 + M3
        # c11 = M0 - M1 + M2 + M6

        # M0
        U[i_off*m + j_off][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[j_off*p + k_off][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[k_off*n + i_off][col] = 1       # c00
        W[(k_off+1)*n + (i_off+1)][col] = 1  # c11
        col += 1

        # M1
        U[(i_off+1)*m + j_off][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[j_off*p + k_off][col] = 1
        W[k_off*n + (i_off+1)][col] = 1   # c10
        W[(k_off+1)*n + (i_off+1)][col] = -1  # c11
        col += 1

        # M2
        U[i_off*m + j_off][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = -1
        W[k_off*n + i_off][col] = 1       # c00
        W[(k_off+1)*n + i_off][col] = 1   # c01
        col += 1

        # M3
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[j_off*p + (k_off+1)][col] = 1
        V[j_off*p + k_off][col] = -1
        W[k_off*n + i_off][col] = 1       # c00
        W[k_off*n + (i_off+1)][col] = 1   # c10
        col += 1

        # M4
        U[i_off*m + j_off][col] = 1
        U[i_off*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[k_off*n + i_off][col] = -1      # c00
        W[(k_off+1)*n + i_off][col] = 1   # c01
        col += 1

        # M5
        U[(i_off+1)*m + j_off][col] = 1
        U[i_off*m + j_off][col] = -1
        V[j_off*p + k_off][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        W[k_off*n + (i_off+1)][col] = 1   # c10
        col += 1

        # M6
        U[i_off*m + (j_off+1)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = -1
        V[j_off*p + (k_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[k_off*n + i_off][col] = -1      # c00
        W[(k_off+1)*n + (i_off+1)][col] = 1  # c11
        col += 1

    # Strassen Block 1: i_off=0, j_off=0, k_off=0
    strassen_block(0, 0, 0)
    # Strassen Block 2: i_off=0, j_off=2, k_off=2
    strassen_block(0, 2, 2)

    # Schoolbook for remaining (i,j,k) not covered by Strassen
    # Strassen covers: i in {0,1}, j in {0,1}, k in {0,1} and i in {0,1}, j in {2,3}, k in {2,3}
    covered = set()
    for i in range(2):
        for j in range(2):
            for k in range(2):
                covered.add((i, j, k))
        for j in range(2, 4):
            for k in range(2, 5):
                covered.add((i, j, k))

    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i*m + j][col] = 1
                    V[j*p + k][col] = 1
                    W[k*n + i][col] = 1
                    col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
