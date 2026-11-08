import time

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = 3, 4, 8
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # Part 1: i=0, schoolbook 1x4x8 = 32 terms
    for j in range(4):
        for k in range(8):
            U[j][col] = 1
            V[j * p + k][col] = 1
            W[k * n][col] = 1
            col += 1

    # Part 2: i=1,2, use 2x4x8 with Strassen on 2x2x2 blocks
    # Strassen 2x2x2 matrices
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]

    for jb in range(2):
        js = jb * 2
        for kb in range(4):
            ks = kb * 2
            for t in range(7):
                for l in range(4):
                    val = strassen_U[t][l]
                    if val:
                        li, lj = l // 2, l % 2
                        gi, gj = 1 + li, js + lj
                        U[gi * m + gj][col] += val
                for l in range(4):
                    val = strassen_V[t][l]
                    if val:
                        lj, lk = l // 2, l % 2
                        gj, gk = js + lj, ks + lk
                        V[gj * p + gk][col] += val
                for l in range(4):
                    val = strassen_W[t][l]
                    if val:
                        li, lk = l // 2, l % 2
                        gi, gk = 1 + li, ks + lk
                        W[gk * n + gi][col] += val
                col += 1

    return {"U": U, "V": V, "W": W}
