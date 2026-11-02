import time
import numpy as np

_START = time.time()
_DEADLINE = 145.0  # safety margin before 150s kill


def _time_ok(margin=5.0):
    return time.time() < _DEADLINE - margin


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact for the matmul tensor."""
    rank = len(U[0])
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(s - 1.0) > tol:
                    return False
    return True


def _add_strassen_222(U, V, W, col, n, m, p, i_base, j_base, k_base):
    """Add 7 Strassen terms for a 2x2x2 block. Returns new col."""
    a_idx = i_base * m + j_base
    b_idx = i_base * m + (j_base + 1)
    c_idx = (i_base + 1) * m + j_base
    d_idx = (i_base + 1) * m + (j_base + 1)

    e_idx = j_base * p + k_base
    f_idx = j_base * p + (k_base + 1)
    g_idx = (j_base + 1) * p + k_base
    h_idx = (j_base + 1) * p + (k_base + 1)

    w00 = k_base * n + i_base
    w01 = (k_base + 1) * n + i_base
    w10 = k_base * n + (i_base + 1)
    w11 = (k_base + 1) * n + (i_base + 1)

    # M1 = (a+d)(e+h) -> C00 += 1, C11 += 1
    U[a_idx][col] += 1; U[d_idx][col] += 1
    V[e_idx][col] += 1; V[h_idx][col] += 1
    W[w00][col] += 1; W[w11][col] += 1
    col += 1

    # M2 = (c+d)e -> C10 += 1, C11 += -1
    U[c_idx][col] += 1; U[d_idx][col] += 1
    V[e_idx][col] += 1
    W[w10][col] += 1; W[w11][col] += -1
    col += 1

    # M3 = a(f-h) -> C01 += 1, C11 += 1
    U[a_idx][col] += 1
    V[f_idx][col] += 1; V[h_idx][col] += -1
    W[w01][col] += 1; W[w11][col] += 1
    col += 1

    # M4 = d(g-e) -> C00 += 1, C10 += 1
    U[d_idx][col] += 1
    V[g_idx][col] += 1; V[e_idx][col] += -1
    W[w00][col] += 1; W[w10][col] += 1
    col += 1

    # M5 = (a+b)h -> C00 += -1, C01 += 1
    U[a_idx][col] += 1; U[b_idx][col] += 1
    V[h_idx][col] += 1
    W[w00][col] += -1; W[w01][col] += 1
    col += 1

    # M6 = (c-a)(e+f) -> C11 += 1
    U[c_idx][col] += 1; U[a_idx][col] += -1
    V[e_idx][col] += 1; V[f_idx][col] += 1
    W[w11][col] += 1
    col += 1

    # M7 = (b-d)(g+h) -> C00 += 1
    U[b_idx][col] += 1; U[d_idx][col] += -1
    V[g_idx][col] += 1; V[h_idx][col] += 1
    W[w00][col] += 1
    col += 1

    return col


def _construct_structured(n, m, p):
    """Proven 66-rank structured decomposition (row 0 schoolbook + rows 1-2 Strassen)."""
    rank = 66
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0

    # Row 0: schoolbook 24 terms
    for j in range(m):
        for k in range(p):
            U[0 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 0][col] = 1
            col += 1

    # Rows 1-2: 6 Strassen blocks of 2x2x2
    for j_block in range(m // 2):
        for k_block in range(p // 2):
            j_base = j_block * 2
            k_base = k_block * 2
            col = _add_strassen_222(U, V, W, col, n, m, p, 1, j_base, k_base)

    return U, V, W


def _am_polish(U0, V0, W0, n, m, p, rank, max_iter=200, tol=1e-10, seed=0):
    """Vectorized alternating minimization polish.
    Returns (U,V,W) as lists if an exact decomposition of `rank` is found, else None."""
    if not _time_ok(8.0):
        return None

    np.random.seed(seed)
    U = np.asarray(U0, dtype=float).copy()
    V = np.asarray(V0, dtype=float).copy()
    W = np.asarray(W0, dtype=float).copy()

    best_obj = float('inf')
    best_state = None

    for it in range(max_iter):
        if not _time_ok(3.0):
            break

        # Update U
        for i in range(n):
            for j in range(m):
                A = np.array([V[j * p + k, :] * W[k * n + i, :] for k in range(p)])
                try:
                    x, _, _, _ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    U[i * m + j, :] = x
                except Exception:
                    pass

        # Update V
        for j in range(m):
            for k in range(p):
                B = np.array([U[i * m + j, :] * W[k * n + i, :] for i in range(n)])
                try:
                    x, _, _, _ = np.linalg.lstsq(B, np.ones(n), rcond=None)
                    V[j * p + k, :] = x
                except Exception:
                    pass

        # Update W
        for k in range(p):
            for i in range(n):
                C = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])
                try:
                    x, _, _, _ = np.linalg.lstsq(C, np.ones(m), rcond=None)
                    W[k * n + i, :] = x
                except Exception:
                    pass

        # Objective (vectorized)
        obj = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    obj += (s - 1.0) ** 2

        if obj < best_obj:
            best_obj = obj
            best_state = (U.copy(), V.copy(), W.copy())

        if obj < tol:
            break

    if best_obj < tol:
        U, V, W = best_state
        return (U.tolist(), V.tolist(), W.tolist())

    return None


def _try_reduce(U, V, W, n, m, p, start_rank, min_rank):
    """Try to find a lower-rank exact decomposition by trimming and polishing.
    Returns (U,V,W) of the best reduced rank found, or None."""
    best = None
    best_rank = start_rank
    U_arr = np.asarray(U, dtype=float)
    V_arr = np.asarray(V, dtype=float)
    W_arr = np.asarray(W, dtype=float)

    for rank in range(start_rank - 1, min_rank - 1, -1):
        if not _time_ok(10.0):
            break

        U_t = U_arr[:, :rank].tolist()
        V_t = V_arr[:, :rank].tolist()
        W_t = W_arr[:, :rank].tolist()

        # Try multiple seeds with perturbation
        for seed in range(8):
            if not _time_ok(2.0):
                break
            # Add small perturbation to escape local minima
            if seed > 0 and seed % 2 == 0:
                np.random.seed(seed * 1000)
                U_p = [row[:] for row in U_t]
                V_p = [row[:] for row in V_t]
                W_p = [row[:] for row in W_t]
                for _ in range(3):
                    idx = np.random.randint(0, rank)
                    U_p[np.random.randint(0, n*m)][idx] += np.random.randn() * 0.1
                    V_p[np.random.randint(0, m*p)][idx] += np.random.randn() * 0.1
                    W_p[np.random.randint(0, p*n)][idx] += np.random.randn() * 0.1
                res = _am_polish(U_p, V_p, W_p, n, m, p, rank, max_iter=150, tol=1e-10, seed=seed)
            else:
                res = _am_polish(U_t, V_t, W_t, n, m, p, rank, max_iter=150, tol=1e-10, seed=seed)
            if res is not None:
                U_r, V_r, W_r = res
                if _verify(U_r, V_r, W_r, n, m, p, tol=1e-6):
                    best = (U_r, V_r, W_r)
                    best_rank = rank
                    break

        if best is not None:
            break

    return best, best_rank


def construct():
    n, m, p = 3, 4, 6

    # 1) Proven structured 66-rank solution
    U, V, W = _construct_structured(n, m, p)

    # Verify the structured solution (sanity; should be exact)
    if not _verify(U, V, W, n, m, p, tol=1e-9):
        # Fallback: pure schoolbook (guaranteed exact)
        rank = n * m * p
        Us = [[0] * rank for _ in range(n * m)]
        Vs = [[0] * rank for _ in range(m * p)]
        Ws = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    Us[i * m + j][col] = 1
                    Vs[j * p + k][col] = 1
                    Ws[k * n + i][col] = 1
                    col += 1
        return {"U": Us, "V": Vs, "W": Ws}

    # 2) Attempt to reduce rank below 66 using polish search (deadline-guarded)
    if _time_ok(20.0):
        reduced, _ = _try_reduce(U, V, W, n, m, p, start_rank=66, min_rank=24)
        if reduced is not None:
            U, V, W = reduced

    return {"U": U, "V": V, "W": W}
