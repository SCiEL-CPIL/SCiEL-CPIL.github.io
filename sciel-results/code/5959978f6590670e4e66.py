"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.perf_counter()
    deadline = 120.0
    n, m, p = DIMENSIONS

    nb = p // 2
    rk = nb * 7

    U = [[0] * rk for _ in range(n * m)]
    V = [[0] * rk for _ in range(m * p)]
    W = [[0] * rk for _ in range(p * n)]

    terms = (
        (1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1),
        (0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, -1),
        (1, 0, 0, 0, 0, 1, 0, -1, 0, 1, 0, 1),
        (0, 0, 0, 1, -1, 0, 1, 0, 1, 0, 1, 0),
        (1, 1, 0, 0, 0, 0, 0, 1, -1, 1, 0, 0),
        (-1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1),
        (0, 1, 0, -1, 0, 0, 1, 1, 1, 0, 0, 0),
    )

    for b in range(nb):
        for j, tm in enumerate(terms):
            c = b * 7 + j
            a0, a1, a2, a3, b0, b1, b2, b3, w0, w1, w2, w3 = tm
            U[0][c] = a0
            U[1][c] = a1
            U[2][c] = a2
            U[3][c] = a3
            V[2 * b][c] = b0
            V[2 * b + 1][c] = b1
            V[p + 2 * b][c] = b2
            V[p + 2 * b + 1][c] = b3
            W[2 * b][c] = w0
            W[2 * b + 1][c] = w1
            W[2 * b + n][c] = w2
            W[2 * b + 1 + n][c] = w3

    if time.perf_counter() - t0 > deadline - 5:
        pass
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
