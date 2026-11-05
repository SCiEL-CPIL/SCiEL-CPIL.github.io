"""MODEL-AND-OPTIMIZE: parameterize job ordering, fit against makespan via local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan_fast(pt, perm):
    """Makespan using the running max of partial column sums (O(n*m) with cache)."""
    n = len(perm)
    m = pt.shape[1]
    cum = np.zeros(m, dtype=np.float64)
    best = 0.0
    for job in perm:
        cum += pt[job]
        val = cum.max()
        if val > best:
            best = val
    return best


def _makespan_fast_tuple(pt, perm):
    """Makespan where perm is a tuple/list of ints."""
    m = pt.shape[1]
    cum = np.zeros(m, dtype=np.float64)
    best = 0.0
    for job in perm:
        cum += pt[job]
        val = cum.max()
        if val > best:
            best = val
    return best


def _neh(pt):
    """NEH heuristic: insert jobs in order of decreasing total processing time."""
    n, m = pt.shape
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    perm = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan_fast_tuple(pt, new_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    return perm


def _palmer(pt):
    """Palmer's heuristic: sort by decreasing priority index."""
    n, m = pt.shape
    # Priority: sum over machines of (m - 2*k) * pt[j,k] where k is machine index (0-based)
    # Actually Palmer's: WSPT-like with weights. Let's use the standard formula.
    # For flow shop, Palmer's index for job j:
    #   sum_{k=1}^{m} (m - 2k) * p_{jk}  ... but indices are 1-based in papers.
    # With 0-based machine index k: weight = (m - 1) - 2*k
    weights = np.array([(m - 1) - 2 * k for k in range(m)], dtype=np.float64)
    priorities = pt @ weights
    perm = np.argsort(-priorities).tolist()
    return perm


def _lpt(pt):
    """LPT: Longest Processing Time first (by total)."""
    totals = pt.sum(axis=1)
    return np.argsort(-totals).tolist()


def _spt(pt):
    """SPT: Shortest Processing Time first (by total)."""
    totals = pt.sum(axis=1)
    return np.argsort(totals).tolist()


def _random_perm(n):
    perm = list(range(n))
    random.shuffle(perm)
    return perm


def _local_search(pt, perm, deadline, max_iters=50000):
    """Iterative improvement with adjacent swaps and insertions."""
    n = len(perm)
    m = pt.shape[1]
    best = _makespan_fast_tuple(pt, perm)
    improved = True
    iters = 0

    while improved and iters < max_iters and time.time() < deadline:
        improved = False
        iters += 1
        # Try adjacent swaps
        for i in range(n - 1):
            if time.time() > deadline:
                return best, perm
            perm2 = perm[:]
            perm2[i], perm2[i + 1] = perm2[i + 1], perm2[i]
            v = _makespan_fast_tuple(pt, perm2)
            if v < best - 1e-9:
                best = v
                perm = perm2
                improved = True
                break
        if improved:
            continue
        # Try insertions
        for i in range(n):
            if time.time() > deadline:
                return best, perm
            item = perm[i]
            for j in range(n):
                if j == i or j == i + 1:
                    continue
                perm2 = perm[:]
                del perm2[i]
                if j > i:
                    j -= 1
                perm2.insert(j, item)
                v = _makespan_fast_tuple(pt, perm2)
                if v < best - 1e-9:
                    best = v
                    perm = perm2
                    improved = True
                    break
            if improved:
                break
    return best, perm


def _segment_reversal(pt, perm, deadline):
    """Try reversing segments to escape local optima."""
    n = len(perm)
    best = _makespan_fast_tuple(pt, perm)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n):
            if time.time() > deadline:
                break
            for j in range(i + 2, n):
                if time.time() > deadline:
                    break
                perm2 = perm[:]
                perm2[i:j + 1] = perm2[i:j + 1][::-1]
                v = _makespan_fast_tuple(pt, perm2)
                if v < best - 1e-9:
                    best = v
                    perm = perm2
                    improved = True
                    break
            if improved:
                break
    return best, perm


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # safety margin under 120s (evaluator kills at 150s)
    pt = np.asarray(processing_times, dtype=np.float64)
    n, m = pt.shape

    if n <= 1:
        return list(range(n))

    # Generate multiple initial solutions
    candidates = []

    # NEH
    candidates.append(_neh(pt))

    # Palmer
    candidates.append(_palmer(pt))

    # LPT
    candidates.append(_lpt(pt))

    # SPT
    candidates.append(_spt(pt))

    # Random starts
    for _ in range(5):
        candidates.append(_random_perm(n))

    # Local search from each candidate
    best_overall = float('inf')
    best_perm_overall = None

    for perm in candidates:
        if time.time() > deadline:
            break
        # Give more time to earlier, higher-quality heuristics
        remaining = deadline - time.time()
        if remaining < 2.0:
            break
        # Allocate time proportional to remaining, but ensure at least some time
        sub_deadline = time.time() + min(remaining * 0.8, 15.0)
        val, perm_ls = _local_search(pt, perm, sub_deadline, max_iters=20000)
        if val < best_overall:
            best_overall = val
            best_perm_overall = perm_ls

    # If we still have time, do segment reversal on the best
    if best_perm_overall is not None and time.time() < deadline - 3.0:
        val, perm_sr = _segment_reversal(pt, best_perm_overall, deadline - 2.0)
        if val < best_overall:
            best_overall = val
            best_perm_overall = perm_sr

    # Final local search pass with remaining time
    if best_perm_overall is not None and time.time() < deadline - 1.0:
        val, perm_final = _local_search(pt, best_perm_overall, deadline - 0.5, max_iters=50000)
        if val < best_overall:
            best_overall = val
            best_perm_overall = perm_final

    if best_perm_overall is None:
        best_perm_overall = list(range(n))

    return best_perm_overall
# EVOLVE-BLOCK-END
