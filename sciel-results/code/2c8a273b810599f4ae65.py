"""Rank-reduction search for the 3x3x5 matrix multiplication tensor."""
import random
import time
import math
import numpy as np

DIMENSIONS = [3, 3, 5]

# Wall-clock guard (evaluator kills at ~150s; leave safety margin).
DEADLINE = 130.0

_U = None
_V = None
_W = None
_SCHOOLBOOK = None


def _schoolbook():
    global _U, _V, _W, _SCHOOLBOOK
    if _SCHOOLBOOK is not None:
        return _U, _V, _W
    n, m, p = DIMENSIONS
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    _U, _V, _W = U, V, W
    _SCHOOLBOOK = True
    return U, V, W


def _tensor_target(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _residual(U, V, W):
    # tensor = U @ V^T  broadcast along W axis
    return np.einsum('ac,bc->ab', U, V) - np.einsum('ac,bc->ab', V, W) * 0.0  # placeholder
    # proper computation below (kept simple for clarity)


def _residual_arr(U, V, W):
    # R[i,j,k] = sum_c U[i,c] V[j,c] W[k,c] ; target T[i,j,k] = (i//m)==(k//n)
    n, m, p = DIMENSIONS
    R = np.einsum('ac,bc,kc->ik', U, V, W)  # shape (n*m, p*n)
    T = np.zeros((n * m, p * n))
    for i in range(n):
        for k in range(p):
            T[i * m + k, k * n + i] = 1.0
    return R - T


def _surrogate(err):
    # smooth objective: total squared error + weight on largest error (epigraph-ish)
    s2 = float(np.sum(err * err))
    emax = float(np.max(np.abs(err)))
    return s2 + 4.0 * emax


def _snap_matrix(A):
    # round to nearest half-integer, keep small
    B = np.round(A * 2.0) / 2.0
    B[np.abs(B) < 1e-9] = 0.0
    return B


def _verify(U, V, W, tol=1e-6):
    n, m, p = DIMENSIONS
    R = np.einsum('ac,bc,kc->ik', U, V, W)
    T = np.zeros((n * m, p * n))
    for i in range(n):
        for k in range(p):
            T[i * m + k, k * n + i] = 1.0
    return float(np.max(np.abs(R - T))) <= tol


def _to_halfint_list(A):
    return [[float(x) for x in row] for row in A]


def _candidate(U0, V0, W0, n, m, p, r, rng, grid):
    # sparse random init: each row gets a few non-zero entries from grid
    def sparse(mat, rows, cols, nzmin, nzmax):
        M = np.zeros((rows, cols))
        for i in range(rows):
            nz = rng.randint(nzmin, nzmax)
            idx = rng.sample(range(cols), min(nz, cols))
            for c in idx:
                M[i, c] = float(rng.choice(grid))
        return M
    return sparse(U0, n * m, r, 2, 5), sparse(V0, m * p, r, 2, 5), sparse(W0, p * n, r, 2, 5)


def _improve(U, V, W, n, m, p, r, rng, grid, deadline, base_err):
    U = U.copy(); V = V.copy(); W = W.copy()
    err = _residual_arr(U, V, W)
    best = _surrogate(err)
    last_improve = time.time()
    stall = 0
    while time.time() < deadline:
        # pick a random factor and a random (row, col) to change
        f = rng.random()
        if f < 1/3:
            M = 'U'; mat = U; rows = n * m
        elif f < 2/3:
            M = 'V'; mat = V; rows = m * p
        else:
            M = 'W'; mat = W; rows = p * n
        row = rng.randrange(rows)
        col = rng.randrange(r)
        old = mat[row, col]
        if old == 0.0:
            # candidate values: 0 and a few small ones
            cands = [0.0] + list(grid[:6])
        else:
            cands = [0.0] + [float(g) for g in grid if abs(g - old) > 1e-9]
            if not cands or len(cands) < 2:
                cands = [0.0, old]
        # find best value for this coordinate
        best_val = old
        best_local = None
        for gv in cands:
            mat[row, col] = gv
            e = _residual_arr(U, V, W)
            s = _surrogate(e)
            if best_local is None or s < best_local:
                best_local = s
                best_val = gv
        mat[row, col] = best_val
        e = _residual_arr(U, V, W)
        s = _surrogate(e)
        if s < best - 1e-9:
            best = s
            last_improve = time.time()
            stall = 0
        else:
            stall += 1
        # reset if long stall
        if stall > 400:
            # kick with a random perturbation of a random row
            f = rng.random()
            if f < 1/3:
                mat, rows = U, n * m
            elif f < 2/3:
                mat, rows = V, m * p
            else:
                mat, rows = W, p * n
            row = rng.randrange(rows)
            for c in range(r):
                if rng.random() < 0.3:
                    mat[row, c] = float(rng.choice(grid))
            stall = 0
        # exactness check
        if best < 1e-7:
            break
    return U, V, W, best


def _search_once(r, n, m, p, rng, deadline):
    grid = [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 1.5, -1.5]
    U0 = np.zeros((n * m, r)); V0 = np.zeros((m * p, r)); W0 = np.zeros((p * n, r))
    U, V, W = _candidate(U0, V0, W0, n, m, p, r, rng, grid)
    # initialize a couple of columns as schoolbook-like to give structure
    U, V, W, score = _improve(U, V, W, n, m, p, r, rng, grid, deadline, float('inf'))
    return U, V, W, score


def construct():
    global _U, _V, _W
    n, m, p = DIMENSIONS
    t_start = time.time()
    deadline = min(t_start + DEADLINE, 120.0 - 2.0)
    # baseline
    sU, sV, sW = _schoolbook()
    best_rank = sU.shape[1]
    best_U, best_V, best_W = sU.copy(), sV.copy(), sW.copy()
    best_score = 0.0  # exact
    rng = random.Random(12345)

    # Try a sequence of ranks below schoolbook, spending limited time each.
    # Known: 3x3x3 rank <= 23, 3x3x4 ~ 30. So target a few values.
    ranks_to_try = [34, 33, 32, 31, 30, 29]
    for r in ranks_to_try:
        if time.time() > deadline - 15:
            break
        per_budget = deadline - 8.0
        per_start = time.time()
        U, V, W, score = _search_once(r, n, m, p, rng, per_start + min(18.0, per_budget - (time.time() - per_start)))
        if score < 1e-7 and _verify(U, V, W):
            # snap to half-integers and re-verify
            Us = _snap_matrix(U); Vs = _snap_matrix(V); Ws = _snap_matrix(W)
            if _verify(Us, Vs, Ws):
                best_rank = r
                best_U, best_V, best_W = Us, Vs, Ws
                break  # found exact at this rank; could try lower but keep safe
        # else keep going

    best_U = _snap_matrix(best_U)
    best_V = _snap_matrix(best_V)
    best_W = _snap_matrix(best_W)
    if not _verify(best_U, best_V, best_W):
        # fall back to guaranteed-valid schoolbook
        sU, sV, sW = _schoolbook()
        best_U, best_V, best_W = sU, sV, sW
        best_rank = sU.shape[1]

    return {
        "U": _to_halfint_list(best_U),
        "V": _to_halfint_list(best_V),
        "W": _to_halfint_list(best_W),
    }
