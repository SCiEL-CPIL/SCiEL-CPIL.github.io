"""Flow-shop makespan minimizer: NEH + CIG constructive, 2/3-opt & swap local search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan(perm, pt):
    """
    Calculate makespan for a permutation using the standard recurrence.
    perm: 1-D array of job indices (length n_jobs)
    pt: (n_jobs, n_machines) processing times
    Returns: float makespan
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    if n_jobs == 0:
        return 0.0
    if n_jobs == 1:
        return float(pt[perm[0]].sum())
    
    # c[j] = completion time of job j on the current machine
    c = np.zeros(n_jobs, dtype=np.float64)
    p = pt[perm]  # (n_jobs, n_machines)
    
    for m in range(n_machines):
        c[0] = p[0, m]
        for j in range(1, n_jobs):
            c[j] = max(c[j], c[j-1]) + p[j, m]
    
    return float(c[-1])


def _neh(processing_times):
    """
    NEH heuristic with fast insertion cost evaluation.
    """
    n_jobs, n_machines = processing_times.shape
    order = list(np.argsort(-processing_times.sum(axis=1)))
    
    perm = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_cost = float('inf')
        
        for pos in range(i + 1):
            # Build candidate permutation
            cand = perm[:pos] + [job] + perm[pos:]
            cost = _makespan(cand, processing_times)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return np.array(perm, dtype=np.int64)


def _p_spt(processing_times):
    """Palmer's SPT rule (shortest processing time on first machine)."""
    return np.argsort(processing_times[:, 0])


def _p_spt_last(processing_times):
    """SPT rule on last machine."""
    return np.argsort(processing_times[:, -1])


def _local_search(perm, pt, deadline, margin):
    """
    Local search with swap, insert, and 2-opt moves.
    Uses fast makespan evaluation.
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    best = perm.copy()
    best_cost = _makespan(best, pt)
    
    improved = True
    
    while improved:
        improved = False
        
        # Swap moves
        for i in range(n_jobs - 1):
            if time.monotonic() > deadline - margin:
                break
            for j in range(i + 1, n_jobs):
                cand = best.copy()
                cand[i], cand[j] = cand[j], cand[i]
                cost = _makespan(cand, pt)
                if cost < best_cost - 1e-9:
                    best_cost = cost
                    best = cand
                    improved = True
                    break
            if improved or time.monotonic() > deadline - margin:
                break
        
        if improved or time.monotonic() > deadline - margin:
            continue
        
        # 2-Opt (segment reversal) moves
        for i in range(n_jobs - 1):
            if time.monotonic() > deadline - margin:
                break
            for j in range(i + 1, n_jobs):
                cand = best.copy()
                cand[i:j+1] = cand[i:j+1][::-1]
                cost = _makespan(cand, pt)
                if cost < best_cost - 1e-9:
                    best_cost = cost
                    best = cand
                    improved = True
                    break
            if improved or time.monotonic() > deadline - margin:
                break
        
        if improved or time.monotonic() > deadline - margin:
            continue
        
        # Insert moves: remove job at i and insert at j
        for i in range(n_jobs):
            if time.monotonic() > deadline - margin:
                break
            for j in range(n_jobs):
                if j == i:
                    continue
                cand = list(best)
                job = cand.pop(i)
                # Adjust j if i < j
                insert_pos = j if j <= i else j - 1
                cand.insert(insert_pos, job)
                cand = np.array(cand, dtype=np.int64)
                cost = _makespan(cand, pt)
                if cost < best_cost - 1e-9:
                    best_cost = cost
                    best = cand
                    improved = True
                    break
            if improved or time.monotonic() > deadline - margin:
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
    deadline = t0 + 145.0
    margin = 5.0

    # 1. NEH heuristic
    best_perm = _neh(processing_times)
    best_cost = _makespan(best_perm, processing_times)
    
    # 2. Local search
    best_perm, best_cost = _local_search(best_perm, processing_times, deadline, margin)
    
    # 3. Try other initial solutions
    # SPT on first machine
    spt_perm = _p_spt(processing_times)
    spt_cost = _makespan(spt_perm, processing_times)
    if spt_cost < best_cost:
        best_perm = spt_perm
        best_cost = spt_cost
        best_perm, best_cost = _local_search(best_perm, processing_times, deadline, margin)
    
    # SPT on last machine
    spt_last_perm = _p_spt_last(processing_times)
    spt_last_cost = _makespan(spt_last_perm, processing_times)
    if spt_last_cost < best_cost:
        best_perm = spt_last_perm
        best_cost = spt_last_cost
        best_perm, best_cost = _local_search(best_perm, processing_times, deadline, margin)
    
    # Random restarts for diversity
    rng = np.random.default_rng(0)
    for _ in range(3):
        if time.monotonic() > deadline - margin - 5.0:
            break
        rand_perm = rng.permutation(n_jobs)
        rand_cost = _makespan(rand_perm, processing_times)
        if rand_cost < best_cost:
            best_perm = rand_perm
            best_cost = rand_cost
            best_perm, best_cost = _local_search(best_perm, processing_times, deadline, margin)
    
    if time.monotonic() > deadline - margin:
        return list(best_perm)

    return list(best_perm)
# EVOLVE-BLOCK-END
