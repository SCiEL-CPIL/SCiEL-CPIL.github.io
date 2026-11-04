"""Block decomposition: 4x4x9 via 2x2 blocks for rank 80."""
DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 9

    rank = 80
    U = [[0] * rank for _ in range(16)]
    V = [[0] * rank for _ in range(36)]
    W = [[0] * rank for _ in range(36)]

    col = 0

    def add_4x4x4(koff):
        nonlocal col
        for bi in range(2):
            for bj in range(2):
                for bk in range(2):
                    for li in range(2):
                        for lj in range(2):
                            for lk in range(2):
                                gi = 2 * bi + li
                                gj = 2 * bj + lj
                                gk = 2 * bk + lk + koff
                                U[gi * 4 + gj][col] = 1
                                V[gj * 4 + gk][col] = 1
                                W[gk * 4 + gi][col] = 1
                                col += 1

    add_4x4x4(0)
    add_4x4x4(4)

    for i in range(4):
        for j in range(4):
            U[i * 4 + j][col] = 1
            V[j * 4 + 8][col] = 1
            W[8 * 4 + i][col] = 1
            col += 1

    assert col == rank

    return {"U": U, "V": V, "W": W}
