"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Group 16 columns into 8 blocks of 2, use Strassen rank-7 per block
    num_blocks = p // 2  # 8 blocks
    rank = 7 * num_blocks  # 56

    U = [[0] * rank for _ in range(n * m)]  # 4 x 56
    V = [[0] * rank for _ in range(m * p)]  # 32 x 56
    W = [[0] * rank for _ in range(p * n)]  # 32 x 56

    for t in range(num_blocks):
        base = 7 * t
        col_e = 2 * t          # V index for B[0][2t]
        col_f = m + 2 * t      # V index for B[1][2t]
        col_g = 2 * t + 1      # V index for B[0][2t+1]
        col_h = m + 2 * t + 1  # V index for B[1][2t+1]

        # W indices: W[k*n + i] where k is output col, i is output row
        w00 = (2 * t) * n + 0      # C[0][2t]
        w10 = (2 * t) * n + 1      # C[1][2t]
        w01 = (2 * t + 1) * n + 0  # C[0][2t+1]
        w11 = (2 * t + 1) * n + 1  # C[1][2t+1]

        # P1 = (a+d)(e+h)
        c = base + 0
        U[0][c] += 1; U[3][c] += 1
        V[col_e][c] += 1; V[col_h][c] += 1
        W[w00][c] += 1; W[w11][c] += 1

        # P2 = (c+d)e
        c = base + 1
        U[2][c] += 1; U[3][c] += 1
        V[col_e][c] += 1
        W[w10][c] += 1; W[w11][c] += -1

        # P3 = a(f-h)
        c = base + 2
        U[0][c] += 1
        V[col_f][c] += 1; V[col_h][c] += -1
        W[w01][c] += 1; W[w11][c] += 1

        # P4 = d(g-e)
        c = base + 3
        U[3][c] += 1
        V[col_g][c] += 1; V[col_e][c] += -1
        W[w00][c] += 1; W[w10][c] += 1

        # P5 = (a+b)h
        c = base + 4
        U[0][c] += 1; U[1][c] += 1
        V[col_h][c] += 1
        W[w00][c] += -1; W[w01][c] += 1

        # P6 = (c-a)(e+f)
        c = base + 5
        U[2][c] += 1; U[0][c] += -1
        V[col_e][c] += 1; V[col_f][c] += 1
        W[w11][c] += 1

        # P7 = (b-d)(g+h)
        c = base + 6
        U[1][c] += 1; U[3][c] += -1
        V[col_g][c] += 1; V[col_h][c] += 1
        W[w00][c] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
