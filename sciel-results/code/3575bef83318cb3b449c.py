import numpy as np
import time

def construct():
    start_time = time.time()
    max_time = 140  # Safety margin before the 150s kill
    n, m, p = 2, 3, 4

    # ------------------------------------------------------------------
    # Proven rank-22 construction
    # ------------------------------------------------------------------
    def build_rank_22():
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val

        # Block 1: Strassen for A[0:2,0:2] * B[0:2,0:2]
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

        # Block 2: Strassen for A[0:2,0:2] * B[0:2,2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

        # Outer product: A[0:2,2:3] * B[2:3,0:4]
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

        return U, V, W, rank

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------
    def verify(U, V, W, rank, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i * m + j][t] * V[j * p + k][t] * W[k * n + i][t]
                    if abs(s - 1.0) > tol:
                        return False
        return True

    def snap_to_half_integer(arr, tol=1e-6):
        snapped = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                nearest_int = round(val)
                if abs(val - nearest_int) < tol:
                    snapped[i, j] = nearest_int
                else:
                    nearest_half = round(val * 2) / 2.0
                    if abs(val - nearest_half) < tol:
                        snapped[i, j] = nearest_half
                    else:
                        snapped[i, j] = val
        return snapped

    def residuals(U, V, W, rank):
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        res = np.empty((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    res[i, j, k] = 1.0 - float(np.dot(U[i * m + j, :rank],
                                                       V[j * p + k, :rank] * W[k * n + i, :rank]))
        return res

    def smooth_penalty(res, scale=50.0):
        a = np.abs(res)
        m = a.max()
        if m < 1e-12:
            return 0.0
        sa = scale * a
        sa_max = sa.max()
        lse = sa_max + np.log(np.sum(np.exp(sa - sa_max)))
        return float(lse / scale)

    # ------------------------------------------------------------------
    # ALS solver
    # ------------------------------------------------------------------
    def _als_solve(U, V, W, rank, max_iter, tol, seed, deadline):
        rng = np.random.RandomState(seed)
        U = np.asarray(U, dtype=float).copy()
        V = np.asarray(V, dtype=float).copy()
        W = np.asarray(W, dtype=float).copy()

        best_pen = np.inf
        best = (U.copy(), V.copy(), W.copy())
        for it in range(max_iter):
            if time.time() > deadline:
                break
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    A = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])
                    b = np.ones(m)
                    try:
                        W_new[k * n + i, :], *_ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        W_new[k * n + i, :] = rng.randn(rank) * 0.1
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    A = np.array([V[j * p + k, :] * W[k * n + i, :] for k in range(p)])
                    b = np.ones(p)
                    try:
                        U_new[i * m + j, :], *_ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        U_new[i * m + j, :] = rng.randn(rank) * 0.1
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    A = np.array([U[i * m + j, :] * W[k * n + i, :] for i in range(n)])
                    b = np.ones(n)
                    try:
                        V_new[j * p + k, :], *_ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        V_new[j * p + k, :] = rng.randn(rank) * 0.1

            diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
            W, U, V = W_new, U_new, V_new

            pen = smooth_penalty(residuals(U, V, W, rank))
            if pen < best_pen:
                best_pen = pen
                best = (U.copy(), V.copy(), W.copy())
            if diff < tol:
                break
        return best, best_pen

    def polish(U, V, W, rank, max_iter=400, tol=1e-15, deadline=None):
        U = np.asarray(U, dtype=float).copy()
        V = np.asarray(V, dtype=float).copy()
        W = np.asarray(W, dtype=float).copy()
        for it in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    A = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])
                    try:
                        W_new[k * n + i, :], *_ = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    except np.linalg.LinAlgError:
                        W_new[k * n + i, :] = W[k * n + i, :]
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    A = np.array([V[j * p + k, :] * W[k * n + i, :] for k in range(p)])
                    try:
                        U_new[i * m + j, :], *_ = np.linalg.lstsq(A, np.ones(p), rcond=None)
                    except np.linalg.LinAlgError:
                        U_new[i * m + j, :] = U[i * m + j, :]
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    A = np.array([U[i * m + j, :] * W[k * n + i, :] for i in range(n)])
                    try:
                        V_new[j * p + k, :], *_ = np.linalg.lstsq(A, np.ones(n), rcond=None)
                    except np.linalg.LinAlgError:
                        V_new[j * p + k, :] = V[j * p + k, :]
            diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
            W, U, V = W_new, U_new, V_new
            if diff < tol:
                break

        def to_lists(arr):
            return [[round(float(x), 12) for x in row] for row in arr]

        U_s = snap_to_half_integer(U); V_s = snap_to_half_integer(V); W_s = snap_to_half_integer(W)
        U_l, V_l, W_l = to_lists(U_s), to_lists(V_s), to_lists(W_s)
        if verify(U_l, V_l, W_l, rank):
            return U_l, V_l, W_l
        U_l, V_l, W_l = to_lists(U), to_lists(V), to_lists(W)
        if verify(U_l, V_l, W_l, rank):
            return U_l, V_l, W_l
        return None

    def try_als(rank, max_iter=250, tol=1e-12, seed=42, init=None, deadline=None):
        if init is not None:
            U0, V0, W0 = init
        else:
            rng = np.random.RandomState(seed)
            U0 = rng.randn(n * m, rank) * 0.5
            V0 = rng.randn(m * p, rank) * 0.5
            W0 = rng.randn(p * n, rank) * 0.5
        (U, V, W), pen = _als_solve(U0, V0, W0, rank, max_iter, tol, seed, deadline)
        U_l = [[round(float(x), 10) for x in row] for row in U]
        V_l = [[round(float(x), 10) for x in row] for row in V]
        W_l = [[round(float(x), 10) for x in row] for row in W]
        if verify(U_l, V_l, W_l, rank):
            return U_l, V_l, W_l
        if pen < 1e-6:
            polished = polish(U, V, W, rank, deadline=deadline)
            if polished is not None:
                return polished
        return None

    # ------------------------------------------------------------------
    # Column-dropping strategy
    # ------------------------------------------------------------------
    def drop_columns_and_refine(cols_to_keep, target_rank, deadline, seed_offset=0):
        """Initialize with rank-22 solution, zero out columns not in cols_to_keep, run ALS."""
        U22, V22, W22, _ = build_rank_22()
        Ua = np.array(U22, dtype=float)
        Va = np.array(V22, dtype=float)
        Wa = np.array(W22, dtype=float)

        # Create initial with only the kept columns
        Ui = Ua[:, cols_to_keep].copy()
        Vi = Va[:, cols_to_keep].copy()
        Wi = Wa[:, cols_to_keep].copy()

        # Add small perturbation
        rng = np.random.RandomState(seed_offset + 42)
        Ui += rng.randn(*Ui.shape) * 0.01
        Vi += rng.randn(*Vi.shape) * 0.01
        Wi += rng.randn(*Wi.shape) * 0.01

        return try_als(target_rank, max_iter=200, seed=seed_offset, init=(Ui, Vi, Wi), deadline=deadline)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------
    best = None
    deadline_search = start_time + max_time - 15

    U22, V22, W22, rank22 = build_rank_22()
    U22_arr = np.array(U22, dtype=float)
    V22_arr = np.array(V22, dtype=float)
    W22_arr = np.array(W22, dtype=float)

    # Try rank 20 by dropping 2 columns from rank-22
    if best is None and time.time() < deadline_search - 30:
        from itertools import combinations
        found = None
        for drop1 in range(22):
            if time.time() > deadline_search:
                break
            for drop2 in range(drop1 + 1, 22):
                keep = [c for c in range(22) if c != drop1 and c != drop2]
                res = drop_columns_and_refine(keep, 20, deadline_search, seed_offset=drop1 * 100 + drop2)
                if res is not None:
                    U_l, V_l, W_l = res
                    if verify(U_l, V_l, W_l, 20):
                        found = (U_l, V_l, W_l, 20)
                        break
            if found is not None:
                break
        if found is not None:
            polished = polish(found[0], found[1], found[2], 20, deadline=start_time + max_time - 5)
            if polished is not None:
                found = (polished[0], polished[1], polished[2], 20)
            best = found

    # Try rank 21 by dropping 1 column from rank-22
    if best is None and time.time() < deadline_search:
        found = None
        for drop in range(22):
            if time.time() > deadline_search:
                break
            keep = [c for c in range(22) if c != drop]
            res = drop_columns_and_refine(keep, 21, deadline_search, seed_offset=drop)
            if res is not None:
                U_l, V_l, W_l = res
                if verify(U_l, V_l, W_l, 21):
                    found = (U_l, V_l, W_l, 21)
                    break
        if found is not None:
            polished = polish(found[0], found[1], found[2], 21, deadline=start_time + max_time - 5)
            if polished is not None:
                found = (polished[0], polished[1], polished[2], 21)
            best = found

    # Try rank 19 with structured initializations
    if best is None and time.time() < deadline_search:
        found = None
        rng = np.random.RandomState(12345)
        for attempt in range(20):
            if time.time() > deadline_search:
                break
            U0 = np.zeros((n * m, 19))
            V0 = np.zeros((m * p, 19))
            W0 = np.zeros((p * n, 19))
            for r in range(19):
                u_density = rng.uniform(0.3, 0.7)
                v_density = rng.uniform(0.3, 0.7)
                w_density = rng.uniform(0.3, 0.7)
                u_mask = rng.rand(n * m) < u_density
                v_mask = rng.rand(m * p) < v_density
                w_mask = rng.rand(p * n) < w_density
                U0[u_mask, r] = rng.randn(u_mask.sum())
                V0[v_mask, r] = rng.randn(v_mask.sum())
                W0[w_mask, r] = rng.randn(w_mask.sum())
            res = try_als(19, max_iter=200, seed=attempt, init=(U0, V0, W0), deadline=deadline_search)
            if res is not None:
                found = (res[0], res[1], res[2], 19)
                break
        if found is not None:
            polished = polish(found[0], found[1], found[2], 19, deadline=start_time + max_time - 5)
            if polished is not None:
                found = (polished[0], polished[1], polished[2], 19)
            best = found

    if best is not None:
        U_l, V_l, W_l, r = best
        return {"U": U_l, "V": V_l, "W": W_l}

    # Fallback: rank-22
    return {"U": U22, "V": V22, "W": W22}
