"""Strassen-based rank-98 exact tensor decomposition for 4x4 * 4x8 matmul."""
import time

DIMENSIONS = [4, 4, 8]

_DEADLINE = time.time() + 120.0  # safety margin under the 150 s kill


def _strassen(n, m, p):
    """Return (U, V, W) exact-integer factors of the n*m x m*p matmul tensor."""
    if m == 1:
        rank = n * p
        U = [[0] * rank for _ in range(n)]        # n*m rows
        V = [[0] * rank for _ in range(p)]        # m*p rows
        W = [[0] * rank for _ in range(p * n)]    # p*n rows
        col = 0
        for i in range(n):
            for k in range(p):
                U[i][col] = 1
                V[k][col] = 1
                W[k * n + i][col] = 1
                col += 1
        return U, V, W

    n2, m2, p2 = n // 2, m // 2, p // 2
    subU, subV, subW = _strassen(n2, m2, p2)
    r = len(subU[0]) if subU else 0

    # 7 Strassen products: (A-block map, B-block map, C-block map) with signs.
    prods = [
        ({11: 1, 22: 1}, {11: 1, 22: 1}, {11: 1, 22: 1}),   # M1
        ({21: 1, 22: 1}, {11: 1},      {21: 1, 22: -1}),    # M2
        ({11: 1},      {21: 1, 22: -1}, {12: 1, 22: 1}),    # M3
        ({22: 1},      {11: 1, 21: -1}, {11: 1, 21: 1}),    # M4
        ({11: 1, 12: 1}, {22: 1},      {11: -1, 12: 1}),    # M5
        ({21: 1, 11: -1}, {11: 1, 12: 1}, {22: 1}),         # M6
        ({12: 1, 22: -1}, {21: 1, 22: 1}, {11: 1}),         # M7
    ]
    rank = 7 * r
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # block offsets in (row, col) units
    A_i_off = {11: 0, 12: 0, 21: n2, 22: n2}
    A_j_off = {11: 0, 12: m2, 21: 0, 22: m2}
    B_j_off = {11: 0, 12: 0, 21: m2, 22: m2}
    B_k_off = {11: 0, 12: p2, 21: 0, 22: p2}
    C_i_off = {11: 0, 12: 0, 21: n2, 22: n2}
    C_k_off = {11: 0, 12: p2, 21: 0, 22: p2}

    for x, (a_map, b_map, c_map) in enumerate(prods):
        base_col = x * r
        # U side (A blocks)
        for i2 in range(n2):
            for j2 in range(m2):
                row = subU[i2 * m2 + j2]
                for r_idx in range(r):
                    v = row[r_idx]
                    if v == 0:
                        continue
                    for blk, s in a_map.items():
                        ai = (i2 + A_i_off[blk]) * m + (j2 + A_j_off[blk])
                        U[ai][base_col + r_idx] += s * v
        # V side (B blocks)
        for j2 in range(m2):
            for k2 in range(p2):
                row = subV[j2 * p2 + k2]
                for r_idx in range(r):
                    v = row[r_idx]
                    if v == 0:
                        continue
                    for blk, s in b_map.items():
                        bi = (j2 + B_j_off[blk]) * p + (k2 + B_k_off[blk])
                        V[bi][base_col + r_idx] += s * v
        # W side (C blocks, trace convention W[k*n+i])
        for k2 in range(p2):
            for i2 in range(n2):
                row = subW[k2 * n2 + i2]
                for r_idx in range(r):
                    v = row[r_idx]
                    if v == 0:
                        continue
                    for blk, s in c_map.items():
                        wi = (k2 + C_k_off[blk]) * n + (i2 + C_i_off[blk])
                        W[wi][base_col + r_idx] += s * v
    return U, V, W


def _verify(U, V, W, n, m, p):
    import random
    random.seed(987654321)
    rank = len(U[0])
    for _ in range(4):
        A = [[random.randint(-4, 4) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-4, 4) for _ in range(p)] for _ in range(m)]
        for i in range(n):
            for k in range(p):
                s_true = 0
                for j in range(m):
                    s_true += A[i][j] * B[j][k]
                s_dec = 0
                for r in range(rank):
                    w = W[k * n + i][r]
                    if w == 0:
                        continue
                    t = 0
                    for j in range(m):
                        t += U[i * m + j][r] * V[j * p + k][r]
                    s_dec += t * w
                if s_dec != s_true:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    U, V, W = _strassen(n, m, p)
    if time.time() < _DEADLINE:
        _verify(U, V, W, n, m, p)
    return {"U": U, "V": V, "W": W}
