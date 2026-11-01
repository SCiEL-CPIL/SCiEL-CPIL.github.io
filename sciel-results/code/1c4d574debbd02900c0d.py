"""Matrix multiplication 2x3 x 3x9 rank decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 3, 9]
_START = time.time()
_DEADLINE = 150.0
_SAFETY = 12.0          # seconds reserved before the kill
_SEARCH_BUDGET = 8.0    # max seconds for the exploratory search


def _time_left():
    return _DEADLINE - _SAFETY - (time.time() - _START)


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _target_tensor(n, m, p):
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T


def _reconstruct(U, V, W):
    n, m, p = DIMENSIONS
    S = np.zeros((n * m, m * p, p * n))
    for c in range(U.shape[1]):
        S += U[:, c][:, None, None] * V[:, c][None, :, None] * W[:, c][None, None, :]
    return S


def _is_valid(U, V, W, T, tol=1e-6):
    S = _reconstruct(U, V, W)
    return bool(np.allclose(S, T, atol=tol))


def _to_list(M):
    out = []
    for row in M:
        out.append([float(x) for x in row])
    return out


def _search(T):
    """Bounded exact search for a rank below the schoolbook value.

    Tests the hypothesis that a sub-54 decomposition exists by trying random
    half-integer factor matrices and verifying exactness. Returns the best
    (U, V, W) found or None. Never allowed to exceed the search budget.
    """
    n, m, p = DIMENSIONS
    base_rank = n * m * p
    best = None
    best_rank = base_rank
    alphabet = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
    rng = np.random.default_rng(12345)

    search_start = time.time()
    for rank in range(base_rank - 1, max(24, base_rank - 10), -1):
        if (time.time() - search_start) >= _SEARCH_BUDGET or _time_left() <= 0:
            break
        max_trials = 400
        for _t in range(max_trials):
            if (time.time() - search_start) >= _SEARCH_BUDGET or _time_left() <= 0:
                break
            U = rng.choice(alphabet, size=(n * m, rank))
            V = rng.choice(alphabet, size=(m * p, rank))
            W = rng.choice(alphabet, size=(p * n, rank))
            if _is_valid(U, V, W, T):
                best = (U, V, W)
                best_rank = rank
                break
    return best, best_rank


def construct():
    n, m, p = DIMENSIONS
    U, V, W = _schoolbook()
    # Attempt a bounded, deadline-guarded search for a lower rank.
    # It can only improve the result; the schoolbook solution is the fallback.
    if _time_left() > _SEARCH_BUDGET + 2.0:
        T = _target_tensor(n, m, p)
        found, _fr = _search(T)
        if found is not None:
            U, V, W = found
    return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
