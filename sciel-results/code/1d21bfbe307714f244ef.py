"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linear_sum_assignment


def _cost(p, a, b):
    return int(np.sum(a * b[np.ix_(p, p)]))


def _lagrangian_init(a, b, deadline):
    """Decompose QAP into a linear assignment on flow + distance penalty,
    solved by subgradient optimization; return the best relaxed permutation."""
    n = len(a)
    t0 = time.time()
    # Center the flow matrix for a symmetric-ish penalty (keeps assignment balanced).
    A = a.astype(np.float64)
    A = A - A.mean()
    B = b.astype(np.float64)
    # Penalty weights (dual variables) on distances, start at the mean flow.
    lam = np.full(n, A.mean(), dtype=np.float64)
    best_p = None
    best_val = np.inf
    # A few subgradient iterations; keep it cheap.
    step_scale = 1.0
    for it in range(12):
        if time.time() > deadline:
            break
        # Relaxed assignment: minimize sum_ij A[i,j] * B[p[i],p[j]] approximated
        # by a linear term using row/col flow sums weighted by lam.
        # Build cost matrix for linear_sum_assignment:
        #   c[i, k] = sum_j A[i,j] * (something for location k)
        # Use: c[i,k] = (A @ 1)[i] * lam[k]  -> pairs heavy-flow facilities
        # with low-distance (low-lam) locations.
        flow_row = A.sum(axis=1)
        dist_col = B.sum(axis=1)
        C = np.outer(flow_row, dist_col)
        C = C - C.mean()
        # Add a small diagonal-free perturbation to break ties stably.
        C = C + 1e-6 * np.abs(np.outer(np.arange(n), np.arange(n) - n))
        r, c = linear_sum_assignment(C)
        p = np.empty(n, dtype=np.int64)
        p[r] = c
        val = _cost(p, a, b)
        if val < best_val:
            best_val = val
            best_p = p.copy()
        # Subgradient update on lam: gradient of the relaxed objective w.r.t. lam
        # is the distance of each location used; push high-lam (heavy) to low-dist.
        g = dist_col[c] - dist_col.mean()
        lam = np.clip(lam - step_scale * g, 0.0, None)
        step_scale *= 0.85
    if best_p is None:
        # Fallback: rank-matching.
        heavy = np.argsort(-a.sum(axis=1), kind="stable")
        central = np.argsort(b.sum(axis=1), kind="stable")
        best_p = np.empty(n, dtype=np.int64)
        best_p[heavy] = central
        best_val = _cost(best_p, a, b)
    return best_p, best_val


def _sa(p, a, b, deadline):
    """Vectorized Simulated Annealing with batched neighborhood evaluation."""
    n = len(p)
    t0 = time.time()
    t_start = 8000.0
    t_end = 0.05
    p = p.astype(np.int64).copy()
    cur = _cost(p, a, b)
    best_p = p.copy()
    best = cur
    # Precompute per-facility flow row for fast delta of a swap.
    A = a.astype(np.float64)
    B = b.astype(np.float64)
    # Number of SA iterations: as many as time allows; batch of candidate swaps.
    BATCH = 256
    # Geometric cooling over a fixed number of iterations; we iterate until deadline.
    n_iter_total = 20000
    for it in range(n_iter_total):
        if time.time() > deadline:
            break
        # Temperature.
        frac = it / max(1, n_iter_total - 1)
        T = t_start * (t_end / t_start) ** frac
        # Generate batch of random distinct swaps.
        idx = np.random.randint(0, n, size=(BATCH, 2))
        # Evaluate delta for each swap (i,j): swap p[i],p[j].
        # delta = A[i,i]B[pj,pj] + A[j,j]B[pi,pi]
        #       + A[i,j]B[pj,pi] + A[j,i]B[pi,pj]
        #       - A[i,i]B[pi,pi] - A[j,j]B[pj,pj]
        #       - A[i,j]B[pi,pj] - A[j,i]B[pj,pi]
        i = idx[:, 0]
        j = idx[:, 1]
        pi = p[i]
        pj = p[j]
        d = (A[i, i] * B[pj, pj] + A[j, j] * B[pi, pi]
             + A[i, j] * B[pj, pi] + A[j, i] * B[pi, pj]
             - A[i, i] * B[pi, pi] - A[j, j] * B[pj, pj]
             - A[i, j] * B[pi, pj] - A[j, i] * B[pj, pi])
        # Accept the best improving or a random Boltzmann-accepted move.
        # Prefer the move with the most negative delta (best improvement).
        k = int(np.argmin(d))
        delta = float(d[k])
        if delta < 0 or (T > 0 and np.random.random() < np.exp(-delta / T)):
            ii = int(i[k]); jj = int(j[k])
            p[ii], p[jj] = p[jj], p[ii]
            cur += delta
            if cur < best:
                best = cur
                best_p = p.copy()
    return best_p, best


def _polish(p, a, b, deadline):
    """Exact 2-opt local search with vectorized delta; repeat until no improvement."""
    n = len(p)
    A = a.astype(np.float64)
    B = b.astype(np.float64)
    p = p.astype(np.int64).copy()
    cur = _cost(p, a, b)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Vectorized delta for all pairs (i<j).
        iu = np.triu_indices(n, k=1)
        i = iu[0]; j = iu[1]
        pi = p[i]; pj = p[j]
        d = (A[i, i] * B[pj, pj] + A[j, j] * B[pi, pi]
             + A[i, j] * B[pj, pi] + A[j, i] * B[pi, pj]
             - A[i, i] * B[pi, pi] - A[j, j] * B[pj, pj]
             - A[i, j] * B[pi, pj] - A[j, i] * B[pj, pi])
        # Apply the best improving swap.
        k = int(np.argmin(d))
        if d[k] < -1e-9:
            ii = int(i[k]); jj = int(j[k])
            p[ii], p[jj] = p[jj], p[ii]
            cur += float(d[k])
            improved = True
    return p, cur


def solve(a, b):
    t0 = time.time()
    n = len(a)
    # Deadline with a safety margin (return best before ~145 s).
    deadline = t0 + 145.0
    # 1) Lagrangian relaxation -> informed initial permutation.
    p, _ = _lagrangian_init(a, b, deadline)
    # 2) Vectorized Simulated Annealing.
    p, _ = _sa(p, a, b, deadline)
    # 3) Exact local-search polish.
    p, _ = _polish(p, a, b, deadline)
    return p.tolist()
# EVOLVE-BLOCK-END
