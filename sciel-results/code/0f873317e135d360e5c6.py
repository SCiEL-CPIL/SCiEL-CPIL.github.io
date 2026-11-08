"""Matrix multiplication 2x4x5 rank-minimal decomposition."""
import time

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    _start = time.time()
    _deadline = _start + 118

    n, m, p = DIMENSIONS
    rank = 36

    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def u_idx(i, j):
        return i * m + j

    def v_idx(j, k):
        return j * p + k

    def w_idx(k, i):
        return k * n + i

    col = 0

    def strassen222(ai, aj, bj, bk, ck, ci):
        nonlocal col
        a00 = u_idx(ai, aj)
        a01 = u_idx(ai, aj + 1)
        a10 = u_idx(ai + 1, aj)
        a11 = u_idx(ai + 1, aj + 1)
        b00 = v_idx(bj, bk)
        b01 = v_idx(bj, bk + 1)
        b10 = v_idx(bj + 1, bk)
        b11 = v_idx(bj + 1, bk + 1)
        c00 = w_idx(ck, ci)
        c01 = w_idx(ck, ci + 1)
        c10 = w_idx(ck + 1, ci)
        c11 = w_idx(ck + 1, ci + 1)

        U[a00][col] = 1; U[a11][col] = 1
        V[b00][col] = 1; V[b11][col] = 1
        W[c00][col] += 1; W[c11][col] += 1
        col += 1
        U[a10][col] = 1; U[a11][col] = 1
        V[b00][col] = 1
        W[c10][col] += 1; W[c11][col] -= 1
        col += 1
        U[a00][col] = 1
        V[b01][col] = 1; V[b11][col] = -1
        W[c01][col] += 1; W[c11][col] += 1
        col += 1
        U[a11][col] = 1
        V[b10][col] = 1; V[b00][col] = -1
        W[c00][col] += 1; W[c10][col] += 1
        col += 1
        U[a00][col] = 1; U[a01][col] = 1
        V[b11][col] = 1
        W[c00][col] -= 1; W[c01][col] += 1
        col += 1
        U[a10][col] = 1; U[a00][col] = -1
        V[b00][col] = 1; V[b01][col] = 1
        W[c11][col] += 1
        col += 1
        U[a01][col] = 1; U[a11][col] = -1
        V[b10][col] = 1; V[b11][col] = 1
        W[c00][col] += 1
        col += 1

    strassen222(0, 0, 0, 0, 0, 0)
    strassen222(0, 2, 2, 0, 0, 0)
    strassen222(0, 0, 0, 2, 2, 0)
    strassen222(0, 2, 2, 2, 2, 0)

    for ai, aj in [(0, 0), (1, 0)]:
        for bj in [0, 1]:
            U[u_idx(ai, bj)][col] = 1
            V[v_idx(bj, 4)][col] = 1
            W[w_idx(4, ai)][col] += 1
            col += 1
    for ai, aj in [(0, 2), (1, 2)]:
        for bj in [2, 3]:
            U[u_idx(ai, aj)][col] = 1
            V[v_idx(bj, 4)][col] = 1
            W[w_idx(4, ai)][col] += 1
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
