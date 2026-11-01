def construct():
    n, m, p = 2, 4, 6

    # Strassen 2x2 @ 2x2 coefficients
    S_U = [
        {0: 1, 3: 1}, {2: 1, 3: 1}, {0: 1}, {3: 1},
        {0: 1, 1: 1}, {2: 1, 0: -1}, {1: 1, 3: -1}
    ]
    S_V = [
        {0: 1, 3: 1}, {0: 1}, {1: 1, 3: -1}, {2: 1, 0: -1},
        {3: 1}, {0: 1, 1: 1}, {2: 1, 3: 1}
    ]
    S_W = [
        {0: 1, 3: 1}, {2: 1, 3: -1}, {1: 1, 3: 1}, {0: 1, 2: 1},
        {0: -1, 1: 1}, {3: 1}, {0: 1}
    ]

    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    for jb in range(3):
        for s in (0, 2):
            for t in range(7):
                for li, sv in S_U[t].items():
                    U[(li // 2) * m + s + li % 2][col] += sv
                for li, sv in S_V[t].items():
                    V[(s + li // 2) * p + 2 * jb + li % 2][col] += sv
                for li, sv in S_W[t].items():
                    W[(2 * jb + li // 2) * n + li % 2][col] += sv
                col += 1

    return {"U": U, "V": V, "W": W}
