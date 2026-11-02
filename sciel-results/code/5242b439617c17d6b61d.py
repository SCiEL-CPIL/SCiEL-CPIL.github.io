import time
import numpy as np

DIMENSIONS = [4, 5, 6]


def _strassen2(terms, n, m, p, Arow, Acol, Brow, Bcol, Crow, Ccol):
    # A is 2x2 at rows Arow..Arow+1, cols Acol..Acol+1
    # B is 2x2 at rows Brow..Brow+1 (== Acol..Acol+1), cols Bcol..Bcol+1
    # C is 2x2 at rows Crow..Crow+1, cols Ccol..Ccol+1
    M = [
        (0, 1, 0, 1),
        (0, 2, 0, 0),
        (0, 0, 1, -1),
        (1, 1, 1, 1),
        (0, 3, 0, 3),
        (2, 3, 1, -3),
        (1, -3, 2, 3),
    ]
    coef = {
        (0, 0): {0: 1, 2: -1, 4: -1, 5: 1},
        (0, 1): {3: 1, 4: 1},
        (1, 0): {1: 1, 2: 1, 5: -1, 6: -1},
        (1, 1): {0: 1, 1: -1, 3: 1, 6: 1},
    }
    for (r, s), terms_map in coef.items():
        for t, cf in terms_map.items():
            va, vb, wa, wb = M[t]
            u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
            u[(Arow + r) * m + (Acol + va)] = cf
            u[(Arow + r) * m + (Acol + vb)] = cf
            v[(Acol + va) * p + (Brow + wa)] = 1.0
            v[(Acol + vb) * p + (Brow + wb)] = 1.0
            w[(Brow + wa) * n + (Crow + r)] = 1.0
            w[(Brow + wb) * n + (Crow + s)] = 1.0
            terms.append((u, v, w))


def _build_structured(n, m, p):
    terms = []
    inner_blocks = [(0, 2), (2, 4), (4, 5)]
    a_row_blocks = [(0, 2), (2, 4)]
    b_col_blocks = [(0, 2), (2, 4), (4, 6)]

    for (ar0, ar1) in a_row_blocks:
        for (in0, in1) in inner_blocks:
            for (bc0, bc1) in b_col_blocks:
                a_h = ar1 - ar0
                in_h = in1 - in0
                b_h = bc1 - bc0
                cr0, cr1 = ar0, ar1
                cc0, cc1 = bc0, bc1
                if a_h == 2 and in_h == 2 and b_h == 2:
                    _strassen2(terms, n, m, p, ar0, in0, in0, bc0, cr0, cc0)
                else:
                    for r in range(a_h):
                        for c in range(in_h):
                            for d in range(b_h):
                                u = np.zeros(n * m); v = np.zeros(m * p); w = np.zeros(p * n)
                                u[(ar0 + r) * m + (in0 + c)] = 1.0
                                v[(in0 + c) * p + (bc0 + d)] = 1.0
                                w[(bc0 + d) * n + (cr0 + r)] = 1.0
                                terms.append((u, v, w))

    rank = len(terms)
    U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
    for t, (u, v, w) in enumerate(terms):
        U[:, t] = u; V[:, t] = v; W[:, t] = w
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-7):
    r = U.shape[1]
    T = np.zeros((n, m, p))
    for rr in range(r):
        A_r = U[:, rr].reshape(n, m)
        B_r = V[:, rr].reshape(m, p)
        C_r = W[:, rr].reshape(p, n)
        for i in range(n):
            for k in range(p):
                T[i, :, k] += A_r[i, :] * B_r[:, k] * C_r[k, i]
    return bool(np.allclose(T, 1.0, atol=tol))


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    U, V, W = _build_structured(n, m, p)
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()

    if not _verify(U, V, W, n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank)); V = np.zeros((m * p, rank)); W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1; V[j * p + k, col] = 1; W[k * n + i, col] = 1
                    col += 1
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_rank = rank

    def try_reduce(U, V, W, target_rank, iters=40):
        U0 = U[:, :target_rank].copy(); V0 = V[:, :target_rank].copy(); W0 = W[:, :target_rank].copy()
        for it in range(iters):
            if time.time() > deadline - 15:
                break
            for i in range(n):
                for j in range(m):
                    M = V[j * p:(j + 1) * p, :].copy()
                    M *= W[np.arange(p) * n + i][:, None]
                    try:
                        U0[i * m + j, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass
            for j in range(m):
                for k in range(p):
                    M = U[np.arange(n) * m + j][:, :].copy()
                    M *= W[k * n + np.arange(n)][:, None]
                    try:
                        V0[j * p + k, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass
            for k in range(p):
                for i in range(n):
                    M = U[i * m + np.arange(m)][:, :].copy()
                    M *= V[np.arange(m) * p + k][:, None]
                    try:
                        W0[k * n + i, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except np.linalg.LinAlgError:
                        pass
            if _verify(U0, V0, W0, n, m, p):
                return U0, V0, W0
        return None

    for target in range(best_rank - 1, max(best_rank - 20, 60), -1):
        if time.time() > deadline - 20:
            break
        res = try_reduce(U, V, W, target)
        if res is not None:
            U, V, W = res
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            best_rank = target

    U_r = np.round(best_U * 2) / 2
    V_r = np.round(best_V * 2) / 2
    W_r = np.round(best_W * 2) / 2
    if _verify(U_r, V_r, W_r, n, m, p):
        best_U, best_V, best_W = U_r, V_r, W_r

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
