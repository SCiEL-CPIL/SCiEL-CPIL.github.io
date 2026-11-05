# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(perm, pt):
    """Compute makespan for a given permutation using the flow-shop recurrence.
    pt: (n, m) array of processing times.
    perm: array of job indices in order.
    Returns makespan (scalar).
    """
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    comp = np.zeros(m, dtype=float)
    for j in range(n):
        job = perm[j]
        for mi in range(m):
            prev_machine = comp[mi - 1] if mi > 0 else 0.0
            same_machine_prev = comp[mi]
            comp[mi] = max(same_machine_prev, prev_machine) + pt[job, mi]
    return comp[-1]

def _makespan_vec(perm, pt):
    """Vectorized makespan computation using numpy."""
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    comp = np.zeros(m, dtype=float)
    for j in range(n):
        job = perm[j]
        pt_row = pt[job]
        for mi in range(m):
            prev = comp[mi - 1] if mi > 0 else 0.0
            comp[mi] = max(comp[mi], prev) + pt_row[mi]
    return comp[-1]

def _neh_initial(pt):
    """NEH heuristic: sort jobs by decreasing total processing time."""
    total_pt = pt.sum(axis=1)
    return np.argsort(-total_pt).astype(int)

def _palmer_initial(pt):
    """Palmer's heuristic: sort by decreasing (n-1-2k)*p_jk + sum of p_jl for l != k."""
    n, m = pt.shape
    if n == 0:
        return np.array([], dtype=int)
    
    # Palmer's key: for each job, compute the key
    # Key = sum over machines of (n - 1 - 2*m) * p[j][m] + sum of p[j][l] for l != m
    # Actually, Palmer's rule: sort by decreasing sum of (n-1-2m)*p_jm
    # Let's use a simpler version: sort by decreasing total processing time
    # But let's try the actual Palmer key
    keys = np.zeros(n)
    for j in range(n):
        for mi in range(m):
            keys[j] += (n - 1 - 2 * mi) * pt[j, mi]
    return np.argsort(-keys).astype(int)

def _lpt_initial(pt):
    """LPT: Longest Processing Time - sort by decreasing total processing time."""
    total_pt = pt.sum(axis=1)
    return np.argsort(-total_pt).astype(int)

def _spt_initial(pt):
    """SPT: Shortest Processing Time - sort by increasing total processing time."""
    total_pt = pt.sum(axis=1)
    return np.argsort(total_pt).astype(int)

def _bottleneck_initial(pt):
    """Bottleneck-based: sort by decreasing processing time on the bottleneck machine."""
    n, m = pt.shape
    if m == 0:
        return np.arange(n)
    # Find the machine with the largest total processing time
    machine_totals = pt.sum(axis=0)
    bottleneck_machine = np.argmax(machine_totals)
    return np.argsort(-pt[:, bottleneck_machine]).astype(int)

def _mean_initial(pt):
    """Sort by decreasing mean processing time."""
    mean_pt = pt.mean(axis=1)
    return np.argsort(-mean_pt).astype(int)

def _first_machine_initial(pt):
    """Sort by decreasing processing time on the first machine."""
    return np.argsort(-pt[:, 0]).astype(int)

def _last_machine_initial(pt):
    """Sort by decreasing processing time on the last machine."""
    return np.argsort(-pt[:, -1]).astype(int)

def _random_initial(n, rng):
    """Random permutation."""
    return rng.permutation(n)

def _local_search_insertion(perm, pt, rng, time_limit=5.0, start_time=None):
    """Insertion local search: try moving each job to a better position."""
    if start_time is None:
        start_time = time.time()
    
    n = len(perm)
    if n <= 1:
        return perm, _makespan(perm, pt)
    
    best_perm = perm.copy()
    best_ms = _makespan(best_perm, pt)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time >= time_limit:
                break
            job = best_perm[i]
            # Try inserting job at every other position
            for j in range(n):
                if j == i:
                    continue
                # Create new permutation by removing job from i and inserting at j
                new_perm = best_perm.copy()
                # Remove job from position i
                remaining = np.delete(new_perm, i)
                if j > i:
                    j -= 1
                # Insert job at position j
                new_perm = np.insert(remaining, j, job)
                ms = _makespan(new_perm, pt)
                if ms < best_ms:
                    best_ms = ms
                    best_perm = new_perm
                    improved = True
                    i = j  # Continue from the new position
                    break
    
    return best_perm, best_ms

def _local_search_reversal(perm, pt, rng, time_limit=5.0, start_time=None):
    """Segment reversal local search: try reversing segments."""
    if start_time is None:
        start_time = time.time()
    
    n = len(perm)
    if n <= 1:
        return perm, _makespan(perm, pt)
    
    best_perm = perm.copy()
    best_ms = _makespan(best_perm, pt)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        for i in range(n):
            if time.time() - start_time >= time_limit:
                break
            for j in range(i + 2, n):
                if time.time() - start_time >= time_limit:
                    break
                # Try reversing segment [i:j+1]
                new_perm = best_perm.copy()
                new_perm[i:j+1] = new_perm[i:j+1][::-1]
                ms = _makespan(new_perm, pt)
                if ms < best_ms:
                    best_ms = ms
                    best_perm = new_perm
                    improved = True
                    break
    
    return best_perm, best_ms

def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on a flow-shop instance."""
    start_time = time.time()
    deadline = 150.0
    safety_margin = 10.0
    effective_deadline = deadline - safety_margin
    
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
    
    rng = np.random.default_rng(42)
    
    # Generate multiple initial solutions using different heuristics
    initial_solutions = []
    
    # NEH
    perm_neh = _neh_initial(processing_times)
    initial_solutions.append(perm_neh)
    
    # Palmer
    perm_palmer = _palmer_initial(processing_times)
    initial_solutions.append(perm_palmer)
    
    # LPT
    perm_lpt = _lpt_initial(processing_times)
    initial_solutions.append(perm_lpt)
    
    # SPT
    perm_spt = _spt_initial(processing_times)
    initial_solutions.append(perm_spt)
    
    # Bottleneck
    perm_bn = _bottleneck_initial(processing_times)
    initial_solutions.append(perm_bn)
    
    # Mean
    perm_mean = _mean_initial(processing_times)
    initial_solutions.append(perm_mean)
    
    # First machine
    perm_first = _first_machine_initial(processing_times)
    initial_solutions.append(perm_first)
    
    # Last machine
    perm_last = _last_machine_initial(processing_times)
    initial_solutions.append(perm_last)
    
    # Add some random solutions
    for _ in range(10):
        perm_rand = _random_initial(n, rng)
        initial_solutions.append(perm_rand)
    
    # Evaluate all initial solutions
    best_perm = None
    best_ms = float('inf')
    
    for perm in initial_solutions:
        ms = _makespan(perm, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm.copy()
    
    # Apply local search to the best initial solution
    best_perm, best_ms = _local_search_insertion(best_perm, processing_times, rng, time_limit=10.0, start_time=start_time)
    
    # Check if we have time for more local search
    if time.time() - start_time < effective_deadline:
        best_perm, best_ms = _local_search_reversal(best_perm, processing_times, rng, time_limit=10.0, start_time=start_time)
    
    # If we still have time, do a second round of local search
    if time.time() - start_time < effective_deadline:
        best_perm, best_ms = _local_search_insertion(best_perm, processing_times, rng, time_limit=15.0, start_time=start_time)
    
    # If we still have time, do another reversal search
    if time.time() - start_time < effective_deadline:
        best_perm, best_ms = _local_search_reversal(best_perm, processing_times, rng, time_limit=15.0, start_time=start_time)
    
    # If we still have time, try a few more random restarts with local search
    remaining_time = effective_deadline - (time.time() - start_time)
    if remaining_time > 10.0:
        num_restarts = min(5, int(remaining_time // 20))
        for _ in range(num_restarts):
            if time.time() - start_time >= effective_deadline:
                break
            perm_rand = _random_initial(n, rng)
            perm_rand, ms_rand = _local_search_insertion(perm_rand, processing_times, rng, time_limit=10.0, start_time=start_time)
            if ms_rand < best_ms:
                best_ms = ms_rand
                best_perm = perm_rand.copy()
    
    return list(best_perm)
# EVOLVE-BLOCK-END
