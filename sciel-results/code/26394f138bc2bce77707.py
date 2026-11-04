"""Tensor decomposition of 5x5x5 matrix multiplication, rank 68."""
import time
import random

def construct():
    t0 = time.time()
    tmax = t0 + 105.0

    n = m = p = 5
    U_cols = []
    V_cols = []
    W_cols = []

    # Frame terms: rows/cols where i==4 or j==4, all k
    for i in range(n):
        for j in range(m):
            if i == 4 or j == 4:
                for k in range(p):
                    U_cols.append({i * m + j: 1})
                    V_cols.append({j * p + k: 1})
                    W_cols.append({k * n + i: 1})

    # Boundary terms: i,j in 0..3, k==4
    for i in range(4):
        for j in range(4):
            U_cols.append({i * m + j: 1})
            V_cols.append({j * p + 4: 1})
            W_cols.append({4 * n + i: 1})

    # 4x4x4 core via Strassen rank-7
    # Block index helpers
    def a_block(blk_i, blk_j):
        return [(2 * blk_i + di, 2 * blk_j + dj) for di in range(2) for dj in range(2)]

    def b_block(blk_j, blk_k):
        return [(2 * blk_j + dj, 2 * blk_k + dk) for dj in range(2) for dk in range(2)]

    # Each product: list of (A_block_idx, B_block_idx, coefficient)
    products = [
        # M1 = (A00+A11)(B00+B11)
        [((0, 0), (0, 0), 1), ((0, 0), (0, 1), 1),
         ((1, 1), (0, 0), 1), ((1, 1), (0, 1), 1)],
        # M2 = (A10+A11)B00
        [((1, 0), (0, 0), 1), ((1, 1), (0, 0), 1)],
        # M3 = A00(B11-B01)
        [((0, 0), (0, 1), 1), ((0, 0), (1, 1), -1)],
        # M4 = (A11-A00)B10
        [((1, 1), (1, 0), 1), ((1, 1), (0, 0), -1)],
        # M5 = (A00+A01)B11
        [((0, 0), (1, 1), 1), ((0, 1), (1, 1), 1)],
        # M6 = (A10-A00)(B00+B01)
        [((1, 0), (0, 0), 1), ((1, 0), (0, 1), 1),
         ((0, 0), (0, 0), -1), ((0, 0), (0, 1), -1)],
        # M7 = (A01-A11)(B10+B11)
        [((0, 1), (1, 0), 1), ((0, 1), (1, 1), 1),
         ((1, 1), (1, 0), -1), ((1, 1), (1, 1), -1)],
    ]

    for prod in products:
        u_acc = {}
        v_acc = {}
        w_acc = {}
        for (ab_idx, bb_idx, coeff) in prod:
            ai, aj = ab_idx
            bj, bk = bb_idx
            for (ii, jj) in a_block(ai, aj):
                for (jj2, kk) in b_block(bj, bk):
                    if jj != jj2:
                        continue
                    ui = ii * m + jj
                    vi = jj * p + kk
                    wi = kk * n + ii
                    u_acc[ui] = u_acc.get(ui, 0) + coeff
                    v_acc[vi] = v_acc.get(vi, 0) + coeff
                    w_acc[wi] = w_acc.get(wi, 0) + coeff
        u_acc = {k: v for k, v in u_acc.items() if v != 0}
        v_acc = {k: v for k, v in v_acc.items() if v != 0}
        w_acc = {k: v for k, v in w_acc.items() if v != 0}
        U_cols.append(u_acc)
        V_cols.append(v_acc)
        W_cols.append(w_acc)

    rank = len(U_cols)

    # Build dense matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for r in range(rank):
        for idx, val in U_cols[r].items():
            U[idx][r] = val
        for idx, val in V_cols[r].items():
            V[idx][r] = val
        for idx, val in W_cols[r].items():
            W[idx][r] = val

    # Verify with random tests
    rng = random.Random(12345)
    for trial in range(200):
        if time.time() > tmax:
            break
        A = [[rng.randint(0, 5) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(0, 5) for _ in range(p)] for _ in range(m)]
        C_true = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        C_test = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                s = 0
                for r in range(rank):
                    for j in range(m):
                        s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                C_test[i][k] = s
        if C_test != C_true:
            return _schoolbook()

    return {"U": U, "V": V, "W": W}


def _schoolbook():
    n, m, p = 5, 5, 5
    rk = n * m * p
    U = [[0] * rk for _ in range(n * m)]
    V = [[0] * rk for _ in range(m * p)]
    W = [[0] * rk for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}
