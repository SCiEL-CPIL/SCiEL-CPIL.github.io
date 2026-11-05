import time
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = 3, 4, 4
    dim_u = n * m          # 12
    dim_v = m * p          # 16
    dim_w = p * n          # 12
    Tsize = dim_u * dim_v * dim_w  # 2304

    start = time.time()
    deadline = start + 112.0

    # ---- Build the rank-44 Strassen decomposition exactly (Fractions) ----
    U = [[Fraction(0)] * 44 for _ in range(dim_u)]
    V = [[Fraction(0)] * 44 for _ in range(dim_v)]
    W = [[Fraction(0)] * 44 for _ in range(dim_w)]

    col = 0
    def add_strassen(Xu, Yv, Zw):
        nonlocal col
        # 7 Strassen products for a 2x2x2 block
        U[Xu[0]][col] += 1; U[Xu[3]][col] += 1
        V[Yv[0]][col] += 1; V[Yv[3]][col] += 1
        W[Zw[0]][col] += 1; W[Zw[3]][col] += 1; col += 1

        U[Xu[2]][col] += 1; U[Xu[3]][col] += 1
        V[Yv[0]][col] += 1
        W[Zw[2]][col] += 1; W[Zw[3]][col] -= 1; col += 1

        U[Xu[0]][col] += 1
        V[Yv[1]][col] += 1; V[Yv[3]][col] -= 1
        W[Zw[1]][col] += 1; W[Zw[3]][col] += 1; col += 1

        U[Xu[3]][col] += 1
        V[Yv[2]][col] += 1; V[Yv[0]][col] -= 1
        W[Zw[0]][col] += 1; W[Zw[2]][col] += 1; col += 1

        U[Xu[0]][col] += 1; U[Xu[1]][col] += 1
        V[Yv[3]][col] += 1
        W[Zw[0]][col] -= 1; W[Zw[1]][col] += 1; col += 1

        U[Xu[2]][col] += 1; U[Xu[0]][col] -= 1
        V[Yv[0]][col] += 1; V[Yv[1]][col] += 1
        W[Zw[3]][col] += 1; col += 1

        U[Xu[1]][col] += 1; U[Xu[3]][col] -= 1
        V[Yv[2]][col] += 1; V[Yv[3]][col] += 1
        W[Zw[0]][col] += 1; col += 1

    add_strassen([0, 1, 4, 5], [0, 1, 4, 5], [0, 3, 1, 4])
    add_strassen([2, 3, 6, 7], [8, 9, 12, 13], [0, 3, 1, 4])
    add_strassen([0, 1, 4, 5], [2, 3, 6, 7], [6, 9, 7, 10])
    add_strassen([2, 3, 6, 7], [10, 11, 14, 15], [6, 9, 7, 10])

    # Schoolbook for the 3rd row of A (1x4x4 blocks)
    for j in range(4):
        U[8 + j][col] += 1; V[j * 4 + 0][col] += 1; W[2][col] += 1; col += 1
        U[8 + j][col] += 1; V[j * 4 + 1][col] += 1; W[5][col] += 1; col += 1
    for j in range(4):
        U[8 + j][col] += 1; V[j * 4 + 2][col] += 1; W[8][col] += 1; col += 1
        U[8 + j][col] += 1; V[j * 4 + 3][col] += 1; W[11][col] += 1; col += 1
    assert col == 44

    # ---- Exact tensor verification ----
    def to_tensor(U, V, W):
        r = len(U[0])
        T = np.zeros((dim_u, dim_v, dim_w), dtype=np.int64)
        for i in range(dim_u):
            ui = U[i]
            for j in range(dim_v):
                vj = V[j]
                for k in range(dim_w):
                    wk = W[k]
                    s = 0
                    for t in range(r):
                        s += int(ui[t]) * int(vj[t]) * int(wk[t])
                    T[i, j, k] = s
        return T

    def expected_tensor():
        T = np.zeros((dim_u, dim_v, dim_w), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i * m + j, j * p + k, k * n + i] = 1
        return T

    def mat_of(U, V, W):
        r = len(U[0])
        M = np.zeros((Tsize, r), dtype=np.int64)
        for t in range(r):
            ucol = np.array([int(x) for x in (U[i][t] for i in range(dim_u))], dtype=np.int64)
            vcol = np.array([int(x) for x in (V[j][t] for j in range(dim_v))], dtype=np.int64)
            wcol = np.array([int(x) for x in (W[k][t] for k in range(dim_w))], dtype=np.int64)
            M[:, t] = np.einsum('i,j,k->ijk', ucol, vcol, wcol).reshape(-1)
        return M

    Texp = expected_tensor()
    U_np = np.array([[int(x) for x in row] for row in U], dtype=np.int64)
    V_np = np.array([[int(x) for x in row] for row in V], dtype=np.int64)
    W_np = np.array([[int(x) for x in row] for row in W], dtype=np.int64)

    if not np.array_equal(to_tensor(U, V, W), Texp):
        # Fallback: schoolbook rank 48 (always valid)
        r = 48
        Uf = np.zeros((dim_u, r), dtype=np.int64)
        Vf = np.zeros((dim_v, r), dtype=np.int64)
        Wf = np.zeros((dim_w, r), dtype=np.int64)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Uf[i * m + j, c] = 1
                    Vf[j * p + k, c] = 1
                    Wf[k * n + i, c] = 1
                    c += 1
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

    # ---- Exact linear-algebra rank reduction ----
    curU, curV, curW = U_np, V_np, W_np
    curM = mat_of(curU, curV, curW)
    curR = curM.shape[1]
    curT = curM.sum(axis=1)
    assert np.array_equal(curT, Texp.reshape(-1))

    def rank_of(M):
        return np.linalg.matrix_rank(M)

    # Phase 1: remove a single redundant term (M[:, r] in span of the rest)
    improved = True
    while improved and curR > 1 and time.time() < deadline:
        improved = False
        for r in range(curR):
            rest = [i for i in range(curR) if i != r]
            if rank_of(curM[:, rest]) == rank_of(curM):
                keep = [i for i in range(curR) if i != r]
                curU = curU[:, keep]
                curV = curV[:, keep]
                curW = curW[:, keep]
                curM = mat_of(curU, curV, curW)
                curR = curM.shape[1]
                improved = True
                break

    # Phase 2: fold term a into term c (T_c <- T_c ± T_a), remove a
    improved = True
    while improved and curR > 1 and time.time() < deadline:
        improved = False
        for a in range(curR):
            for c in range(curR):
                if a == c:
                    continue
                for sgn in (1, -1):
                    Mnew = curM.copy()
                    Mnew[:, c] = Mnew[:, c] + sgn * Mnew[:, a]
                    keep = [i for i in range(curR) if i != a]
                    Mred = Mnew[:, keep]
                    if rank_of(Mred) == rank_of(Mnew):
                        # apply
                        curU = curU.copy(); curV = curV.copy(); curW = curW.copy()
                        curU[:, c] = curU[:, c] + sgn * curU[:, a]
                        curV[:, c] = curV[:, c] + sgn * curV[:, a]
                        curW[:, c] = curW[:, c] + sgn * curW[:, a]
                        curU = curU[:, keep]
                        curV = curV[:, keep]
                        curW = curW[:, keep]
                        curM = mat_of(curU, curV, curW)
                        curR = curM.shape[1]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Phase 3: split off a new term (T_c <- T_c - T_a, add new T_a), reduce rank by 1
    improved = True
    while improved and curR > 1 and time.time() < deadline:
        improved = False
        for a in range(curR):
            for c in range(curR):
                if a == c:
                    continue
                for sgn in (1, -1):
                    Mnew = curM.copy()
                    Mnew[:, c] = Mnew[:, c] - sgn * Mnew[:, a]
                    Mnew = np.column_stack([Mnew, sgn * curM[:, a]])
                    if rank_of(Mnew) == rank_of(curM):
                        curU = curU.copy(); curV = curV.copy(); curW = curW.copy()
                        curU[:, c] = curU[:, c] - sgn * curU[:, a]
                        curV[:, c] = curV[:, c] - sgn * curV[:, a]
                        curW[:, c] = curW[:, c] - sgn * curW[:, a]
                        curU = np.column_stack([curU, sgn * curU[:, a]])
                        curV = np.column_stack([curV, sgn * curV[:, a]])
                        curW = np.column_stack([curW, sgn * curW[:, a]])
                        curM = mat_of(curU, curV, curW)
                        curR = curM.shape[1]
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

    # Final exact verification
    if not np.array_equal(curM.sum(axis=1), Texp.reshape(-1)):
        r = 48
        Uf = np.zeros((dim_u, r), dtype=np.int64)
        Vf = np.zeros((dim_v, r), dtype=np.int64)
        Wf = np.zeros((dim_w, r), dtype=np.int64)
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Uf[i * m + j, c] = 1
                    Vf[j * p + k, c] = 1
                    Wf[k * n + i, c] = 1
                    c += 1
        return {"U": Uf.tolist(), "V": Vf.tolist(), "W": Wf.tolist()}

    return {"U": curU.tolist(), "V": curV.tolist(), "W": curW.tolist()}
