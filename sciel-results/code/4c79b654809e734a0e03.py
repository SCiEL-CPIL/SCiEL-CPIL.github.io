"""Matrix multiplication 4x4x5 — randomized rank-reduction search.

Strategy:
  * Start from the guaranteed schoolbook decomposition (rank 80).
  * Randomly generate low-rank candidates (integer / half-integer entries).
  * For each candidate, form the induced tensor T = U V^T W^T (einsum) and
    compare it EXACTLY to the true multiplication tensor T_true.
  * Keep the lowest rank found.  A safety deadline guard returns the best
    valid result before the wall-clock limit.
"""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]


def _true_tensor():
    """Build the exact multiplication tensor T_true[i, j, k] = delta(i+k, j)."""
    n, m, p = DIMENSIONS
    T = np.zeros((n, m, p), dtype=np.int64)
    for i in range(n):
        for k in range(p):
            j = i + k
            if 0 <= j < m:
                T[i, j, k] = 1
    return T


def _candidate(U, V, W):
    """Induced tensor T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]."""
    return np.einsum('ijc,jkc,kic->ijk', U, V, W)


def _half_or_int(rng, p_half):
    """Sample an integer (p) or half-integer (1-p) from a small set."""
    vals = [-2, -1, 0, 1, 2]
    if rng.random() < p_half:
        return rng.choice(vals) + 0.5
    return rng.choice(vals)


def construct():
    start = time.time()
    deadline = start + 140.0   # safety margin before the 150 s kill
    n, m, p = DIMENSIONS
    T_true = _true_tensor()
    flat_true = T_true.reshape(-1)

    rng = random.Random(12345)
    np_rng = np.random.default_rng(12345)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    U0, V0, W0 = schoolbook()
    best_rank = U0.shape[1]
    best = (U0, V0, W0)
    # sanity: schoolbook must reproduce the tensor exactly
    assert np.array_equal(_candidate(U0, V0, W0).reshape(-1), flat_true)

    # descending rank ladder to try
    for rank in range(75, 0, -1):
        if time.time() > deadline:
            break
        # how many random trials at this rank before giving up on this rank
        # (more trials at higher rank, fewer at very low rank)
        trials = max(20, 3000 // max(1, rank))
        for t in range(trials):
            if time.time() > deadline:
                break
            # mixture: mostly small integer entries, some half-integers,
            # occasional larger magnitudes for diversity
            p_half = rng.random() * 0.4
            mag = rng.choice([1, 1, 2, 3])
            U = np.array([[_half_or_int(rng, p_half) * rng.choice([-1, 1])
                           if rng.random() < 0.85 else 0
                           for _ in range(rank)] for _ in range(n * m)])
            V = np.array([[_half_or_int(rng, p_half) * rng.choice([-1, 1])
                           if rng.random() < 0.85 else 0
                           for _ in range(rank)] for _ in range(m * p)])
            W = np.array([[_half_or_int(rng, p_half) * rng.choice([-1, 1])
                           if rng.random() < 0.85 else 0
                           for _ in range(rank)] for _ in range(p * n)])
            # scale a random subset by mag to broaden the search
            for M in (U, V, W):
                idx = np_rng.choice(M.shape[0], size=max(1, M.shape[0] // 4),
                                    replace=False)
                M[idx, :] *= mag
            T = _candidate(U, V, W)
            if np.array_equal(T.reshape(-1), flat_true):
                best_rank = rank
                best = (U.copy(), V.copy(), W.copy())
                # found a valid low rank; continue trying lower ranks
                break

    U, V, W = best
    return {
        "U": [[float(x) for x in row] for row in U],
        "V": [[float(x) for x in row] for row in V],
        "W": [[float(x) for x in row] for row in W],
    }
