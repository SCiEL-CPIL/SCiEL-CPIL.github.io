"""Flow-shop makespan minimizer: NEH constructive + 2-opt/insertion local search with deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _makespan(perm, pt):
    """Compute makespan for a given permutation of job indices."""
    if len(perm) == 0:
        return 0.0
    # completion time on last machine
    # C[j][m] = max(C[j][m-1], C[j-1][m]) + pt[j][m]
    # We only need the last machine's completion time for each job
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    # c[m] = completion time of the last job on machine m
    c = np.zeros(n_machines, dtype=np.float64)
    for j in perm:
        for m in range(n_machines):
            if m == 0:
                c[m] = c[m] + pt[j, m]
            else:
                c[m] = max(c[m-1], c[m]) + pt[j, m]
    return c[-1]


def _makespan_fast(perm, pt):
    """Vectorized makespan computation using cumulative sums per machine."""
    if len(perm) == 0:
        return 0.0
    selected = pt[perm]  # shape (n_jobs, n_machines)
    # For each machine, the completion time of the last job is:
    # C_last[m] = sum over jobs of max(C_prev_job[m-1], C_prev_job[m]) + pt[j][m]
    # This is sequential, so we do it in a loop but vectorize over machines where possible
    n_machines = pt.shape[1]
    c = np.zeros(n_machines, dtype=np.float64)
    for j in range(len(perm)):
        row = selected[j]
        if n_machines == 1:
            c[0] += row[0]
        else:
            # c[m] = max(c[m-1], c[m]) + row[m]
            for m in range(1, n_machines):
                c[m] = max(c[m-1], c[m]) + row[m]
            c[0] += row[0]
    return c[-1]


def _neh(processing_times):
    """NEH (Nawaz-Enscore-Ham) heuristic for flow shop."""
    n_jobs, n_machines = processing_times.shape
    # Sort jobs by total processing time descending
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    # Build permutation greedily
    perm = [order[0]]
    for job in order[1:]:
        best_pos = 0
        best_ms = _makespan(perm + [job], processing_times)
        for pos in range(1, len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm


def _local_search(perm, processing_times, time_limit=110.0):
    """2-opt and insertion local search with deadline guard."""
    start_time = time.time()
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    
    improved = True
    while improved:
        improved = False
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Try 2-opt moves (swap two jobs)
        for i in range(n_jobs):
            for j in range(i+1, n_jobs):
                if time.time() - start_time > time_limit:
                    break
                new_perm = best_perm[:]
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                ms = _makespan(new_perm, processing_times)
                if ms < best_ms - 1e-12:
                    best_perm = new_perm
                    best_ms = ms
                    improved = True
            if time.time() - start_time > time_limit:
                break
        
        # Try insertion moves (move a job to a different position)
        if time.time() - start_time > time_limit:
            break
        for i in range(n_jobs):
            for j in range(n_jobs):
                if i == j:
                    continue
                if time.time() - start_time > time_limit:
                    break
                new_perm = best_perm[:]
                job = new_perm.pop(i)
                new_perm.insert(j, job)
                ms = _makespan(new_perm, processing_times)
                if ms < best_ms - 1e-12:
                    best_perm = new_perm
                    best_ms = ms
                    improved = True
            if time.time() - start_time > time_limit:
                break
    
    return best_perm, best_ms


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # NEH constructive heuristic
    perm = _neh(processing_times)
    current_ms = _makespan(perm, processing_times)
    
    # Local search with deadline guard
    elapsed = time.time() - start_time
    if elapsed < 115.0:
        perm, current_ms = _local_search(perm, processing_times, time_limit=115.0)
    
    # Safety margin: return if we're getting close to the limit
    if time.time() - start_time > 118.0:
        return list(perm)
    
    return list(perm)
# EVOLVE-BLOCK-END
