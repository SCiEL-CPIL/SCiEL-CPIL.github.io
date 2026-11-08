"""Flow-shop makespan minimizer: NEH + CIG constructive, 2/3-opt & swap local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan(perm, pt):
    """Makespan of a job permutation. O(m) via cumulative sums."""
    c = 0.0
    for m in range(pt.shape[1]):
        c += perm[m] if False else 0.0  # placeholder (unused)
    # proper: last machine cumulative sum
    return float(np.cumsum(perm, axis=0)[-1])


def _makespan_fast(perm, pt):
    """Fast makespan: last machine cumulative sum of the selected rows."""
    return float(pt[perm].sum(axis=0)[-1])


def _insert_cost(perm, pt, job, pos, n_jobs):
    """Cost of inserting job at position pos into perm (length n_jobs-1)."""
    # returns makespan of the new sequence
    new = np.empty(n_jobs, dtype=np.int64)
    if pos == 0:
        new[0] = job
        new[1:] = perm
    elif pos == n_jobs - 1:
        new[:-1] = perm
        new[-1] = job
    else:
        new[:pos] = perm[:pos]
        new[pos] = job
        new[pos + 1:] = perm[pos:]
    return _makespan_fast(new, pt)


def _neh(processing_times):
    """NEH heuristic: order jobs by total processing time desc, insert each optimally."""
    n_jobs, n_machines = processing_times.shape
    order = list(np.argsort(-processing_times.sum(axis=1)))
    perm = np.empty(n_jobs, dtype=np.int64)
    perm[0] = order[0]
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_cost = float('inf')
        for pos in range(i + 1):
            # build candidate
            cand = np.empty(i + 1, dtype=np.int64)
            if pos == 0:
                cand[0] = job
                cand[1:] = perm[:i]
            elif pos == i:
                cand[:i] = perm[:i]
                cand[i] = job
            else:
                cand[:pos] = perm[:pos]
                cand[pos] = job
                cand[pos + 1:] = perm[pos:i]
            cost = float(processing_times[cand].sum(axis=0)[-1])
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        # commit
        new = np.empty(i + 1, dtype=np.int64)
        if best_pos == 0:
            new[0] = job
            new[1:] = perm[:i]
        elif best_pos == i:
            new[:i] = perm[:i]
            new[i] = job
        else:
            new[:best_pos] = perm[:best_pos]
            new[best_pos] = job
            new[best_pos + 1:] = perm[best_pos:i]
        perm[:i + 1] = new
    return perm


def _p_spt(processing_times):
    """Palmer's SPT rule (shortest processing time on first machine)."""
    return np.argsort(processing_times[:, 0])


def _local_search(perm, processing_times, deadline, margin):
    """2-opt + swap + 3-opt (insert) best-improvement local search with time guard."""
    n_jobs, n_machines = processing_times.shape
    best = perm.copy()
    best_cost = _makespan_fast(best, processing_times)
    improved = True
    while improved:
        improved = False
        # 2-opt: reverse a segment
        for i in range(n_jobs - 1):
            if time.monotonic() > deadline - margin:
                break
            for j in range(i + 2, n_jobs):
                cand = best.copy()
                cand[i:j + 1] = cand[i:j + 1][::-1]
                cost = _makespan_fast(cand, processing_times)
                if cost < best_cost - 1e-12:
                    best_cost = cost
                    best = cand
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        # swap: exchange two positions
        for i in range(n_jobs):
            if time.monotonic() > deadline - margin:
                break
            for j in range(i + 1, n_jobs):
                cand = best.copy()
                cand[i], cand[j] = cand[j], cand[i]
                cost = _makespan_fast(cand, processing_times)
                if cost < best_cost - 1e-12:
                    best_cost = cost
                    best = cand
                    improved = True
                    break
            if improved:
                break
        if improved:
            continue
        # 3-opt: move a job to a new position
        for i in range(n_jobs):
            if time.monotonic() > deadline - margin:
                break
            job = best[i]
            for j in range(n_jobs):
                if j == i or j == i + 1:
                    continue
                cand = np.delete(best, i)
                # insert at j (in the reduced array, adjust index)
                insert_at = j if j > i else j
                cand = np.insert(cand, insert_at, job)
                cost = _makespan_fast(cand, processing_times)
                if cost < best_cost - 1e-12:
                    best_cost = cost
                    best = cand
                    improved = True
                    break
            if improved:
                break
    return best, best_cost


def order_jobs(processing_times):
    processing_times = np.asarray(processing_times, dtype=float)
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    t0 = time.monotonic()
    deadline = t0 + 115.0  # safety margin vs 150s kill
    margin = 3.0

    # Constructive candidates
    cands = []
    cands.append(_neh(processing_times))
    cands.append(_p_spt(processing_times))
    # random restarts for diversity (small count)
    rng = np.random.default_rng(0)
    for _ in range(3):
        cands.append(rng.permutation(n_jobs))

    best = cands[0]
    best_cost = _makespan_fast(best, processing_times)
    for c in cands[1:]:
        cost = _makespan_fast(c, processing_times)
        if cost < best_cost:
            best_cost = cost
            best = c

    # Local search on the best, then on others if time remains
    best, best_cost = _local_search(best, processing_times, deadline, margin)

    # If time remains, try local search on other constructive starts
    for c in cands:
        if time.monotonic() > deadline - margin - 5.0:
            break
        if np.array_equal(c, best):
            continue
        # quick: local search from this start
        cand_best, cand_cost = _local_search(c, processing_times, deadline, margin)
        if cand_cost < best_cost:
            best_cost = cand_cost
            best = cand_best

    if time.monotonic() > deadline - margin:
        return list(best)

    return list(best)
# EVOLVE-BLOCK-END
