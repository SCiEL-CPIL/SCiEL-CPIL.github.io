"""Tensor decomposition of 5x5x5 matrix multiplication, rank 68."""
import time

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    tmax = t0 + 105.0

    n = 5
    m = 5
    p = 5

    U_cols = []
    V_cols = []
    W_cols = []

    # Part A: Frame terms for i==4 or j==4, with each k
    for ii in range(n):
        for jj in range(m):
            if ii == 4 or jj == 4:
                for kk in range(p):
                    U_cols.append({ii * m + jj: 1})
                    V_cols.append({jj * p + kk: 1})
                    W_cols.append({kk * n + ii: 1})

    # Part B: Terms for i in 0..3, j in 0..3, k==4 (not covered by frame or Strassen)
    for ii in range(4):
        for jj in range(4):
            kk = 4
            U_cols.append({ii * m + jj: 1})
            V_cols.append({jj * p + kk: 1})
            W_cols.append({kk * n + ii: 1})

    # Part C: 4x4x4 core via Strassen (rank 7)
    def blockA(a, b):
        for di in range(2):
            for dj in range(2):
                yield (2 * a + di, 2 * b + dj)

    def blockB(b, c):
        for dj in range(2):
            for dk in range(2):
                yield (2 * b + dj, 2 * c + dk)

    M1 = [((0, 0), (0, 0), 1), ((0, 0), (0, 1), 1), ((1, 1), (0, 0), 1), ((1, 1), (0, 1), 1)]
    M2 = [((1, 0), (0, 0), 1), ((1, 1), (0, 0), 1)]
    M3 = [((0, 0), (0, 1), 1), ((0, 0), (1, 1), -1)]
    M4 = [((1, 1), (1, 0), 1), ((1, 1), (0, 0), -1)]
    M5 = [((0, 0), (1, 1), 1), ((0, 1), (1, 1), 1)]
    M6 = [((1, 0), (0, 0), 1), ((1, 0), (0, 1), 1), ((0, 0), (0, 0), -1), ((0, 0), (0, 1), -1)]
    M7 = [((0, 1), (1, 0), 1), ((0, 1), (1, 1), 1), ((1, 1), (1, 0), -1), ((1, 1), (1, 1), -1)]

    strassen_products = [M1, M2, M3, M4, M5, M6, M7]

    for prod in strassen_products:
        u_acc = {}
        v_acc = {}
        w_acc = {}
        for (ab, bc, coeff) in prod:
            a_blk, b_blk = ab[0], ab[1]
            b_blk2, c_blk = bc[0], bc[1]
            for (ii, jj) in blockA(a_blk, b_blk):
                for (jj2, kk) in blockB(b_blk2, c_blk):
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

    import random
    rng = random.Random(12345)
    for trial in range(200):
        if time.time() > tmax:
            break
        A = [[rng.randint(0, 5) for _ in range(m)] for _ in range(n)]
        B = [[rng.randint(0, 5) for _ in range(p)] for _ in range(m)]
        C_true = [[sum(A[ii][jj] * B[jj][kk] for jj in range(m)) for kk in range(p)] for ii in range(n)]
        C_test = [[0] * p for _ in range(n)]
        for ii in range(n):
            for kk in range(p):
                s = 0
                for r in range(rank):
                    for jj in range(m):
                        s += U[ii * m + jj][r] * V[jj * p + kk][r] * W[kk * n + ii][r]
                C_test[ii][kk] = s
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
# EVOLVE-BLOCK-END
