"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6

    # 2x2x2 Strassen rank-7 decomposition
    S_U = [
        [1, 0, 1, 0, 1, -1, 0],   # (0,0)
        [0, 0, 0, 0, 1, 0, 1],    # (0,1)
        [0, 1, 0, 0, 0, 1, 0],    # (1,0)
        [1, 1, 0, 1, 0, 0, -1],   # (1,1)
    ]
    S_V = [
        [1, 1, 0, -1, 0, 1, 0],   # (0,0)
        [0, 0, 1, 0, 0, 1, 0],    # (0,1)
        [0, 0, 0, 1, 0, 0, 1],    # (1,0)
        [1, 0, -1, 0, 1, 0, 1],   # (1,1)
    ]
    S_W = [
        [1, 0, 0, 1, -1, 0, 1],   # (0,0)
        [0, 1, 0, 1, 0, 0, 0],    # (0,1)
        [0, 0, 1, 0, 1, 0, 0],    # (1,0)
        [1, -1, 1, 0, 0, 1, 0],   # (1,1)
    ]

    # Block sizes: 2, 2, 2
    # We have n=4, m=4, p=6
    # n blocks: 2 (each of size 2)
    # m blocks: 2 (each of size 2)
    # p blocks: 3 (each of size 2)
    # Total rank = 2 * 2 * 3 * 7 = 84

    n_blocks = 2
    m_blocks = 2
    p_blocks = 3
    bs = 2

    total_rank = n_blocks * m_blocks * p_blocks * 7

    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]

    col_offset = 0
    for bi in range(n_blocks):
        for bj in range(m_blocks):
            for bk in range(p_blocks):
                for c in range(7):
                    col = col_offset + c

                    # U rows: (i, j)
                    for di in range(bs):
                        for dj in range(bs):
                            i = bi * bs + di
                            j = bj * bs + dj
                            U[i * m + j][col] = S_U[di * bs + dj][c]

                    # V rows: (j, k)
                    for dj in range(bs):
                        for dk in range(bs):
                            j = bj * bs + dj
                            k = bk * bs + dk
                            V[j * p + k][col] = S_V[dj * bs + dk][c]

                    # W rows: (k, i)
                    for dk in range(bs):
                        for di in range(bs):
                            k = bk * bs + dk
                            i = bi * bs + di
                            W[k * n + i][col] = S_W[dk * bs + di][c]

                col_offset += 7

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
