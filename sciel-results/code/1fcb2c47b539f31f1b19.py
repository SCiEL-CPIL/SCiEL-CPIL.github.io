"""Flow-shop permutation heuristic with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(processing_times, perm):
    """Compute makespan for a given permutation (list or array of job indices)."""
    # processing_times: (n_jobs, n_machines)
    # perm: sequence of job indices in processing order
    n_machines = processing_times.shape[1]
    # C[m] = completion time on machine m for the last job processed
    C = np.zeros(n_machines, dtype=np.float64)
    for job in perm:
        for m in range(n_machines):
            C[m] = max(C[m], C[m-1] if m > 0 else 0.0) + processing_times[job, m]
    return C[-1]


def _makespan_fast(processing_times, perm):
    """Faster makespan using numpy where possible for small n_machines."""
    # For very small n_machines, the loop is fine.
    # We'll stick with the loop for clarity and correctness.
    n_machines = processing_times.shape[1]
    C = np.zeros(n_machines, dtype=np.float64)
    for job in perm:
        for m in range(n_machines):
            if m == 0:
                C[m] = C[m] + processing_times[job, m]
            else:
                C[m] = max(C[m], C[m-1]) + processing_times[job, m]
    return C[-1]


def _neh(processing_times):
    """NEH heuristic: order by decreasing total processing time, then insert each job into best position."""
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    # Sort jobs by decreasing total processing time
    order = np.argsort(-totals).tolist()
    
    # Start with first job
    current_perm = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(current_perm) + 1):
            new_perm = current_perm[:pos] + [job] + current_perm[pos:]
            ms = _makespan(processing_times, new_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
    
    return current_perm


def _palmer(processing_times):
    """Palmer's dispatching rule: sort by decreasing (n-1)*sum + sum of (n-1-2j)*t_ij."""
    n_jobs, n_machines = processing_times.shape
    n = n_machines
    scores = np.zeros(n_jobs)
    for j in range(n_jobs):
        s = 0.0
        for m in range(n_machines):
            s += (n - 1 - 2 * m) * processing_times[j, m]
        scores[j] = s
    return list(np.argsort(-scores))


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    
    Args:
        processing_times: np.ndarray of shape (n_jobs, n_machines)
    
    Returns:
        list of job indices representing the permutation
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    processing_times = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Generate initial solutions from multiple heuristics
    candidates = []
    
    # 1. NEH heuristic
    neh_perm = _neh(processing_times)
    candidates.append(neh_perm)
    
    # 2. Palmer rule
    palmer_perm = _palmer(processing_times)
    candidates.append(palmer_perm)
    
    # 3. Longest total processing time first (original seed)
    lpt_perm = list(np.argsort(-processing_times.sum(axis=1)))
    candidates.append(lpt_perm)
    
    # 4. Shortest total processing time first
    spt_perm = list(np.argsort(processing_times.sum(axis=1)))
    candidates.append(spt_perm)
    
    # Evaluate all candidates
    best_perm = candidates[0]
    best_ms = _makespan(processing_times, best_perm)
    
    for perm in candidates[1:]:
        ms = _makespan(processing_times, perm)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # If we have time, do local search (2-opt / adjacent swap)
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0:  # At least 5s left
        # Local search with adjacent swaps
        improved = True
        while improved and (time.time() - start_time) < deadline - 2.0:
            improved = False
            for i in range(len(best_perm) - 1):
                if (time.time() - start_time) >= deadline - 2.0:
                    break
                # Try swapping adjacent jobs i and i+1
                new_perm = best_perm.copy()
                new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
                ms = _makespan(processing_times, new_perm)
                if ms < best_ms:
                    best_ms = ms
                    best_perm = new_perm
                    improved = True
    
    # If still have time, try 2-opt style: reverse a segment
    elapsed = time.time() - start_time
    if elapsed < deadline - 5.0 and n_jobs <= 200:
        for _ in range(50):  # Limited iterations
            if (time.time() - start_time) >= deadline - 2.0:
                break
            i = random.randint(0, n_jobs - 2)
            j = random.randint(i + 1, n_jobs - 1)
            new_perm = best_perm[:i] + list(reversed(best_perm[i:j+1])) + best_perm[j+1:]
            ms = _makespan(processing_times, new_perm)
            if ms < best_ms:
                best_ms = ms
                best_perm = new_perm
    
    return list(best_perm)
# EVOLVE-BLOCK-END
