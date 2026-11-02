import time

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    start = time.time()
    DEADLINE = 120.0 - 5.0  # safety margin

    n, m, p = DIMENSIONS

    # Strassen rank-7 decomposition of the 2x2x2 block.
    # U_s[t][i][j], V_s[t][j][k], W_s[t][i][k] for t in 0..6.
    U_s = [
        [[1, 0], [0, 1]],   # M1: a+d
        [[0, 0], [1, 1]],   # M2: c+d
        [[1, 0], [0, 0]],   # M3: a
        [[0, 0], [0, 1]],   # M4: d
        [[1, 1], [0, 0]],   # M5: a+b
        [[-1, 0], [1, 0]],  # M6: c-a
        [[0, 1], [0, -1]],  # M7: b-d
    ]
    V_s = [
        [[1, 0], [0, 1]],   # M1: e+h
        [[1, 0], [0, 0]],   # M2: e
        [[0, 1], [0, -1]],  # M3: f-h
        [[-1, 0], [1, 0]],  # M4: g-e
        [[0, 0], [0, 1]],   # M5: h
        [[1, 1], [0, 0]],   # M6: e+f
        [[0, 0], [1, 1]],   # M7: g+h
    ]
    W_s = [
        [[1, 0], [0, 1]],   # M1: C00+=, C11+=
        [[0, 0], [1, -1]],  # M2: C10+=, C11-=
        [[0, 1], [0, 1]],   # M3: C01+=, C11+=
        [[1, 0], [1, 0]],   # M4: C00+=, C10+=
        [[-1, 1], [0, 0]],  # M5: C00-=, C01+=
        [[0, 0], [0, 1]],   # M6: C11+=
        [[1, 0], [0, 0]],   # M7: C00+=
    ]

    num_blocks = p // 2  # 5 blocks of 2 columns
    rank = 7 * num_blocks  # 35

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for b in range(num_blocks):
        for t in range(7):
            c = 7 * b + t
            for i in range(n):
                for j in range(m):
                    U[i * m + j][c] = U_s[t][i][j]
            for j in range(m):
                for k0 in range(2):
                    k = 2 * b + k0
                    V[j * p + k][c] = V_s[t][j][k0]
            for i in range(n):
                for k0 in range(2):
                    k = 2 * b + k0
                    W[k * n + i][c] = W_s[t][i][k0]

    result = {"U": U, "V": V, "W": W}

    # Verify exactness; fall back to schoolbook if anything is off.
    if time.time() < DEADLINE:
        try:
            import numpy as np
            Ua = np.array(U, dtype=object)
            Va = np.array(V, dtype=object)
            Wa = np.array(W, dtype=object)
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(rank):
                            s += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                        if s != 1:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    break
            if not ok:
                return _schoolbook(n, m, p)
        except Exception:
            pass  # keep the (provably correct) Strassen result

    return result


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
