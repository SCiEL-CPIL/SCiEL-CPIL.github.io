"""Flow-shop permutation: NEH init + vectorized makespan + iterated 2-opt local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan_vectorized(pt, order):
    """Compute makespan for a job permutation using vectorized recurrence.

    C[i][j] = max(C[i-1][j], C[i][j-1]) + p[i][j]
    Makespan = C[n-1][m-1].
    """
    if order.size == 0:
        return 0.0
    n = order.size
    m = pt.shape[1]
    if m == 0:
        return 0.0
    # Matrix of processing times in the given order: shape (n, m)
    P = pt[order]
    # Row-wise cumulative sum along machines: S[i][j] = sum_{k<=j} P[i][k]
    row_csum = np.cumsum(P, axis=1)
    # C[i][j] = max_{k<=i} row_csum[k][j] + S[i-1][j-1]  (with S[-1][*]=0)
    # Build prefix-max of row_csum along jobs.
    prefix_max = np.maximum.accumulate(row_csum, axis=0)  # shape (n, m)
    # shift row_csum by one row, zero on top: prev[j] = S[i-1][j-1]
    if n > 1:
        prev = np.empty_like(row_csum)
        prev[0] = 0.0
        prev[1:] = row_csum[:-1, :-1] if m > 1 else 0.0
    else:
        prev = np.zeros_like(row_csum)
    C = prefix_max + prev
    return float(C[-1, -1])


def _neh_rule(pt):
    """Nawaz-Enscore-Ham constructive heuristic."""
    n = pt.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    m = pt.shape[1]
    # forward and backward cumulative sums
    fwd = np.cumsum(pt, axis=1)          # fwd[i][j] = sum_{k<=j} p[i][k]
    bwd = np.cumsum(pt[:, ::-1], axis=1)[:, ::-1]  # bwd[i][j] = sum_{k>=j} p[i][k]
    # weight = sum over j of fwd[i][j] * bwd[i][j]
    weights = np.sum(fwd * bwd, axis=1)
    order = []
    remaining = list(range(n))
    for _ in range(n):
        # pick job with max weight among remaining
        best = -1
        best_w = -np.inf
        for j in remaining:
            if weights[j] > best_w:
                best_w = weights[j]
                best = j
        order.append(best)
        remaining.remove(best)
        # update weights of remaining: subtract contribution of the placed job
        # contribution of job j at position k (0-indexed) is fwd[j][k]*bwd[j][k]
        # We placed best at position len(order)-1
        pos = len(order) - 1
        for j in remaining:
            weights[j] -= fwd[j, pos] * bwd[j, pos]
    return np.array(order, dtype=int)


def _two_opt_eval(pt, order, i, j):
    """Makespan after reversing segment [i, j] (inclusive)."""
    new_order = np.concatenate([order[:i], order[i:j + 1][::-1], order[j + 1:]])
    return _makespan_vectorized(pt, new_order)


def order_jobs(processing_times):
    pt = np.asarray(processing_times, dtype=float)
    n = pt.shape[0]
    if n == 0:
        return []
    if n == 1:
        return [0]
    m = pt.shape[1]
    if m == 0:
        return list(range(n))

    start = time.time()
    deadline = start + 150.0
    safety = 10.0  # safety margin in seconds

    # Initial solution via NEH
    best_order = _neh_rule(pt)
    best_ms = _makespan_vectorized(pt, best_order)

    # Iterated Local Search with 2-opt
    rng = np.random.default_rng(12345)
    improved = True
    while improved and (time.time() + safety) < deadline:
        improved = False
        # Local search: iterate 2-opt until no improvement
        changed = True
        while changed and (time.time() + safety) < deadline:
            changed = False
            # Sample 2-opt moves; for small n do all, for large n sample
            if n <= 60:
                pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
                # shuffle for diversity
                pairs = list(pairs)
                rng.shuffle(pairs)
            else:
                # sample a bounded number of random 2-opt pairs
                num_samples = min(2000, n * n // 2)
                pairs = []
                for _ in range(num_samples):
                    i = int(rng.integers(0, n))
                    j = int(rng.integers(i + 1, n))
                    pairs.append((i, j))
            for (i, j) in pairs:
                if (time.time() + safety) >= deadline:
                    break
                ms = _two_opt_eval(pt, best_order, i, j)
                if ms < best_ms - 1e-12:
                    best_ms = ms
                    best_order = np.concatenate([best_order[:i], best_order[i:j + 1][::-1], best_order[j + 1:]])
                    changed = True
                    improved = True
        # If no improvement in this local search pass, perturb to escape local optimum
        if not improved and (time.time() + safety) < deadline:
            # Perturbation: reverse a random segment (a k-opt style kick)
            k = int(rng.integers(2, min(6, n)))
            i = int(rng.integers(0, n - k + 1))
            j = i + k - 1
            perturbed = np.concatenate([best_order[:i], best_order[i:j + 1][::-1], best_order[j + 1:]])
            pm = _makespan_vectorized(pt, perturbed)
            # Accept perturbation (iterated local search typically accepts to escape)
            best_order = perturbed
            best_ms = pm

    return list(np.asarray(best_order, dtype=int))
# EVOLVE-BLOCK-END
