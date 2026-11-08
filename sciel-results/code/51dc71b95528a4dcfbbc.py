"""Flow-shop makespan minimizer: NEH + Correct Local Search."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _calc_makespan(perm, pt):
    """
    Calculate the makespan for a given permutation.
    perm: array of job indices
    pt: processing times (n_jobs, n_machines)
    Returns: makespan (float)
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    if n_jobs == 0:
        return 0.0
    
    # Select processing times for the jobs in the permutation order
    # p[j, m] is processing time of job perm[j] on machine m
    p = pt[perm]
    
    # c[j] stores the completion time of job j on the current machine
    c = np.zeros(n_jobs, dtype=float)
    
    for m in range(n_machines):
        for j in range(n_jobs):
            if j == 0:
                # First job on this machine: starts at 0 (or after previous machine, but c[0] is 0 for m=0)
                # Wait, c[j] currently holds completion time on machine m-1.
                # For m=0, c is zeros, so start is 0.
                # For m>0, c[j] is completion on m-1.
                start_time = c[j]
            else:
                # Must wait for previous job on this machine to finish
                # c[j-1] has already been updated to completion on machine m
                start_time = max(c[j], c[j-1])
            
            c[j] = start_time + p[j, m]
            
    return c[-1]


def _neh(processing_times):
    """
    NEH heuristic: order jobs by total processing time desc, insert each optimally.
    """
    n_jobs, n_machines = processing_times.shape
    order = list(np.argsort(-processing_times.sum(axis=1)))
    
    perm = np.empty(n_jobs, dtype=np.int64)
    perm[0] = order[0]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_cost = float('inf')
        
        # Try inserting job at each position in the current partial permutation
        # Current partial perm is perm[0:i]
        for pos in range(i + 1):
            # Construct candidate permutation of length i+1
            # This is O(i) per pos, so O(i^2) per job, O(n^3) total.
            # For n=100, 10^6 ops. Acceptable.
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
            
            cost = _calc_makespan(cand, processing_times)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        
        # Commit the best insertion
        new_perm = np.empty(i + 1, dtype=np.int64)
        if best_pos == 0:
            new_perm[0] = job
            new_perm[1:] = perm[:i]
        elif best_pos == i:
            new_perm[:i] = perm[:i]
            new_perm[i] = job
        else:
            new_perm[:best_pos] = perm[:best_pos]
            new_perm[best_pos] = job
            new_perm[best_pos + 1:] = perm[best_pos:i]
        
        perm[:i + 1] = new_perm
        
    return perm


def _local_search(perm, processing_times, deadline, margin):
    """
    Local search using 2-opt (segment reversal) and swap.
    """
    n_jobs, n_machines = processing_times.shape
    best = perm.copy()
    best_cost = _calc_makespan(best, processing_times)
    improved = True
    
    while improved:
        improved = False
        
        # 2-Opt: Reverse a segment [i, j]
        for i in range(n_jobs - 1):
            if time.monotonic() > deadline - margin:
                return best, best_cost
            for j in range(i + 1, n_jobs):
                cand = best.copy()
                cand[i:j + 1] = cand[i:j + 1][::-1]
                cost = _calc_makespan(cand, processing_times)
                if cost < best_cost - 1e-12:
                    best_cost = cost
                    best = cand
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Swap: Exchange two positions i, j
        for i in range(n_jobs):
            if time.monotonic() > deadline - margin:
                return best, best_cost
            for j in range(i + 1, n_jobs):
                cand = best.copy()
                cand[i], cand[j] = cand[j], cand[i]
                cost = _calc_makespan(cand, processing_times)
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
    deadline = t0 + 145.0  # Safety margin
    margin = 5.0

    # 1. Constructive Heuristic: NEH
    best_perm = _neh(processing_times)
    best_cost = _calc_makespan(best_perm, processing_times)
    
    # 2. Local Search
    best_perm, best_cost = _local_search(best_perm, processing_times, deadline, margin)
    
    # 3. If time remains, try a few random restarts or other heuristics
    # Simple SPT on first machine
    spt_perm = np.argsort(processing_times[:, 0])
    spt_cost = _calc_makespan(spt_perm, processing_times)
    if spt_cost < best_cost:
        best_perm = spt_perm
        best_cost = spt_cost
        best_perm, best_cost = _local_search(best_perm, processing_times, deadline, margin)
        
    # Random restarts
    rng = np.random.default_rng(42)
    for _ in range(5):
        if time.monotonic() > deadline - margin:
            break
        rand_perm = rng.permutation(n_jobs)
        rand_cost = _calc_makespan(rand_perm, processing_times)
        if rand_cost < best_cost:
            best_perm = rand_perm
            best_cost = rand_cost
            best_perm, best_cost = _local_search(best_perm, processing_times, deadline, margin)
            
    return list(best_perm)
# EVOLVE-BLOCK-END
