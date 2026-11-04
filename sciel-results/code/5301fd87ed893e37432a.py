"""QAPLIB wil100 solver: rank-matching multi-start + vectorized O(n^2) delta local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _cost(p, b, w):
    """Exact integer cost = sum_{i,j} a[i,j] * b[p[i],p[j]], using symmetric w."""
    # cost = 0.5 * sum_{i,j} w[i,j] * b[p[i],p[j]]
    return int(round(0.5 * np.sum(w[np.arange(len(w))[:, None], np.arange(len(w))] * b[p][:, p])))


def _starts(n, rng, a, b):
    """Yield a few diverse initial permutations."""
    # Rank-matching: heavy flow rows -> central (short-distance) location rows.
    heavy = np.argsort(-a.sum(axis=1), kind="stable")
    central = np.argsort(b.sum(axis=1), kind="stable")
    p0 = np.empty(n, dtype=np.int64)
    p0[heavy] = central
    yield p0.copy()

    # Reverse rank-matching.
    p1 = np.empty(n, dtype=np.int64)
    p1[heavy] = central[::-1]
    yield p1.copy()

    # A few random starts for diversity.
    for _ in range(3):
        yield rng.permutation(n).astype(np.int64)


def _full_scan(p, W, B, iters=40):
    """Full-scan best-improvement using vectorized delta. Returns (p, improved)."""
    n = len(p)
    improved = False
    for _ in range(iters):
        best_i, best_j, best_d = -1, -1, 0
        for i in range(n - 1):
            # delta for swapping i with each j > i
            diff = B[p[i]] - B[p[i + 1:]]          # B[p[j]] - B[p[i]]
            d = (W[i] - W[i + 1:]) @ diff          # (W_i - W_j).(B_pj - B_pi)
            idx = np.argmin(d)
            if d[idx] < best_d:
                best_d = d[idx]
                best_i = i
                best_j = i + 1 + idx
        if best_i < 0:
            break
        p[best_i], p[best_j] = p[best_j], p[best_i]
        improved = True
    return p, improved


def _random_hill(p, W, B, budget, rng):
    """Cheap random-sampling hill climb until budget seconds elapse."""
    n = len(p)
    t_end = time.time() + budget
    while time.time() < t_end:
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n))
        if i == j:
            continue
        d = (W[i] - W[j]) @ (B[p[j]] - B[p[i]])
        if d < 0:
            p[i], p[j] = p[j], p[i]
    return p


def _polish(p, W, B, iters=8):
    """High-precision final polish: full-scan best-improvement with more iterations."""
    return _full_scan(p, W, B, iters=iters)[0]


def solve(a, b):
    n = len(a)
    t0 = time.time()
    DEADLINE = 150.0
    MARGIN = 10.0  # safety margin before the 150 s kill

    # Symmetric weight matrix w[i,j] = a[i,j] + a[j,i]; cost = 0.5*sum w[i,j] b[p[i],p[j]].
    w = a + a.T
    W = np.ascontiguousarray(w, dtype=np.int64)   # W[i] = w[i,:]
    B = np.ascontiguousarray(b, dtype=np.int64)   # B[k] = b[:,k]

    rng = np.random.default_rng(12345)

    best_p = None
    best_c = None

    def remaining():
        return DEADLINE - MARGIN - (time.time() - t0)

    starts = list(_starts(n, rng, a, b))
    for k, p in enumerate(starts):
        if remaining() <= 2.0:
            break
        p = p.copy()
        # Phase 1: full-scan best-improvement.
        p, _ = _full_scan(p, W, B, iters=30)
        # Phase 2: cheap random hill climb to escape / refine.
        p = _random_hill(p, W, B, budget=2.0, rng=rng)
        # Phase 3: high-precision polish.
        p = _polish(p, W, B, iters=8)
        c = _cost(p, b, w)
        if best_c is None or c < best_c:
            best_c = c
            best_p = p.copy()
        if remaining() <= 2.0:
            break

    # Final polish on the global best if time remains.
    if best_p is not None and remaining() > 1.0:
        best_p = _polish(best_p, W, B, iters=10)
        best_c = _cost(best_p, b, w)

    if best_p is None:
        # Fallback: return a valid permutation.
        best_p = np.arange(n, dtype=np.int64)

    return best_p.tolist()
# EVOLVE-BLOCK-END
