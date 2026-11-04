"""Different strategy: rank-deficient randomized tensor power iteration (ALS)
initialized from a rank-98 recursive-Strassen seed, searching for a shared
decomposition of the 4x4x8 multiplication tensor (rank < 98)."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = 130  # safety margin before 150s kill


def _time_left():
    return _DEADLINE - (time.time() - _start_time)


def _build_strassen_4x4x4():
    """Exact rank-49 recursive-Strassen decomposition of 4x4x4."""
    n, m, p = 4, 4, 4
    rank = 49
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    strassen_2x2 = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): -1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 1): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]
    high_level_terms = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}),
    ]
    hl_c_contrib = {
        0: [(0, 0, 1), (1, 1, 1)],
        1: [(1, 0, 1), (1, 1, -1)],
        2: [(0, 1, 1), (1, 1, 1)],
        3: [(0, 0, 1), (1, 0, 1)],
        4: [(0, 0, -1), (0, 1), (0, 1, 1)] if False else [(0, 0, -1), (0, 1, 1)],
        5: [(1, 1, 1)],
        6: [(0, 0, 1)],
    }

    col = 0
    for hl in range(7):
        A_hl_dict, B_hl_dict = high_level_terms[hl]
        for sub in range(7):
            A_sub_dict, B_sub_dict, C_sub_dict = strassen_2x2[sub]
            U_term = np.zeros((4, 4))
            for (br, bc), a_coeff in A_sub_dict.items():
                hl_coeff = A_hl_dict.get((br, bc), 0)
                total = a_coeff * hl_coeff
                if abs(total) > 1e-10:
                    for r in range(2 * br, 2 * br + 2):
                        for c in range(2 * bc, 2 * bc + 2):
                            U_term[r, c] += total
            V_term = np.zeros((4, 4))
            for (br, bc), b_coeff in B_sub_dict.items():
                hl_coeff = B_hl_dict.get((br, bc), 0)
                total = b_coeff * hl_coeff
                if abs(total) > 1e-10:
                    for r in range(2 * br, 2 * br + 2):
                        for c in range(2 * bc, 2 * bc + 2):
                            V_term[r, c] += total
            W_term = np.zeros((4, 4))
            for (cr, cc), c_sub in C_sub_dict.items():
                for (cbr, cbc, c_hl) in hl_c_contrib[hl]:
                    total = c_sub * c_hl
                    if abs(total) > 1e-10:
                        W_term[cbr * 2 + cr, cbc * 2 + cc] += total
            U[col * 4:(col + 1) * 4, col] = U_term.flatten()
            V[col * 4:(col + 1) * 4, col] = V_term.flatten()
            W[col * 4:(col + 1) * 4, col] = W_term.flatten()
            col += 1
    return U, V, W


def _build_strassen_4x4x8():
    """Exact rank-98 decomposition of 4x4x8 (two independent 4x4x4 blocks)."""
    n, m, p = 4, 4, 8
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    U4, V4, W4 = _build_strassen_4x4x4()
    U[:, 0:49] = U4
    V[0:16, 0:49] = V4
    W[0:16, 0:49] = W4
    U[:, 49:98] = U4
    V[16:32, 49:98] = V4
    W[16:32, 49:98] = W4
    return U, V, W


def _error(U, V, W, n, m, p):
    r = U.shape[1]
    U3 = U.reshape(n, m, r)
    V3 = V.reshape(m, p, r)
    W3 = W.reshape(p, n, r)
    T = np.einsum('ijt,jkt,kit->ijk', U3, V3, W3)
    return float(np.max(np.abs(T - 1.0)))


def _als_sweep(U, V, W, n, m, p):
    """One alternating-least-squares sweep (rank-deficient compatible)."""
    r = U.shape[1]
    # update U: U[i*m+j,:] solves (V[j*p:(j+1)*p,:] * W[arange(p)*n + i,:]) x = 1
    for i in range(n):
        for j in range(m):
            A = V[j * p:(j + 1) * p, :] * W[np.arange(p) * n + i, :]
            try:
                U[i * m + j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
            except Exception:
                pass
    # update V
    for j in range(m):
        for k in range(p):
            A = U[np.arange(n) * m + j, :] * W[k * n + np.arange(n), :]
            try:
                V[j * p + k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
            except Exception:
                pass
    # update W
    for k in range(p):
        for i in range(n):
            A = U[i * m + np.arange(m), :] * V[np.arange(m) * p + k, :]
            try:
                W[k * n + i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
            except Exception:
                pass


def _refine(U, V, W, n, m, p, iters):
    for _ in range(iters):
        if _time_left() < 15:
            break
        _als_sweep(U, V, W, n, m, p)


def _try_rank(r, n, m, p, seed, max_iter):
    """Randomized rank-deficient ALS from a random rank-r init."""
    rng = np.random.default_rng(seed)
    U = rng.standard_normal((n * m, r))
    V = rng.standard_normal((m * p, r))
    W = rng.standard_normal((p * n, r))
    best = None
    best_err = np.inf
    for it in range(max_iter):
        if _time_left() < 15:
            break
        _als_sweep(U, V, W, n, m, p)
        err = _error(U, V, W, n, m, p)
        if err < best_err:
            best_err = err
            best = (U.copy(), V.copy(), W.copy(), err)
        if best_err < 1e-7:
            break
        # occasional random restart of a few columns to escape bad local minima
        if it % 25 == 24:
            for _ in range(r // 8 + 1):
                c = rng.integers(0, r)
                U[:, c] = rng.standard_normal(n * m)
                V[:, c] = rng.standard_normal(m * p)
                W[:, c] = rng.standard_normal(p * n)
    return best


def construct():
    n, m, p = DIMENSIONS  # 4, 4, 8
    schoolbook_rank = n * m * p  # 128

    # schoolbook fallback
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1.0
                V_school[j * p + k, col] = 1.0
                W_school[k * n + i, col] = 1.0
                col += 1
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = schoolbook_rank

    # rank-98 recursive Strassen (exact, verified)
    try:
        U98, V98, W98 = _build_strassen_4x4x8()
        if _error(U98, V98, W98, n, m, p) < 1e-7:
            best_U, best_V, best_W = U98, V98, W98
            best_rank = 98
    except Exception:
        pass

    # --- NEW: rank-deficient randomized ALS searching for rank < 98 ---
    # Target ranks to attempt (below 98). For each, try a few seeds.
    target_ranks = [90, 84, 80, 76, 72, 68, 64]
    for r in target_ranks:
        if _time_left() < 40:
            break
        if r >= best_rank:
            continue
        found = False
        for seed in [0, 1, 2]:
            if _time_left() < 30:
                break
            res = _try_rank(r, n, m, p, seed, max_iter=300)
            if res is not None and res[3] < 1e-7:
                best_U, best_V, best_W = res[0], res[1], res[2]
                best_rank = r
                found = True
                break
        if found:
            continue  # try next (smaller) rank
        # if no exact solution at this rank, keep going to smaller ranks

    # --- NEW: try to compress the rank-98 Strassen seed to lower rank ---
    # Use the rank-98 factors as a warm start and run ALS at reduced ranks.
    try:
        U98, V98, W98 = _build_strassen_4x4x8()
        for r in [96, 94, 92, 90, 88]:
            if _time_left() < 40 or r >= best_rank:
                break
            U = U98[:, :r].copy()
            V = V98[:, :r].copy()
            W = W98[:, :r].copy()
            _refine(U, V, W, n, m, p, iters=200)
            err = _error(U, V, W, n, m, p)
            if err < 1e-7:
                best_U, best_V, best_W = U, V, W
                best_rank = r
    except Exception:
        pass

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
