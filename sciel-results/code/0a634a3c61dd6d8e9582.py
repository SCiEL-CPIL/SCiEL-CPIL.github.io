"""Improved flow-shop scheduling with NEH + 2-opt local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(order, p):
    """Compute makespan for a given job order on a flow-shop."""
    n = len(order)
    if n == 0:
        return 0.0
    # completion[m][j] = completion time of job j on machine m
    # We process jobs in the given order
    comp = np.zeros((p.shape[1], n), dtype=np.float64)
    for j in range(n):
        job = order[j]
        for m in range(p.shape[1]):
            if m == 0:
                comp[m, j] = p[job, m]
            else:
                comp[m, j] = max(comp[m-1, j], comp[m, j-1]) + p[job, m]
    return comp[-1, -1]


def _neh_order(p, rng):
    """
    NEH heuristic (Nawaz, Enscore, Ham, 1983).
    1. Sort jobs in non-increasing order of total processing time.
    2. Insert each job into the best position in the partial schedule.
    """
    n, m = p.shape
    # Sort by total processing time descending
    totals = p.sum(axis=1)
    order = list(np.argsort(-totals, kind='stable'))
    
    # Build schedule by insertion
    schedule = []
    for job in order:
        # Try all positions
        best_pos = 0
        best_ms = np.inf
        for pos in range(len(schedule) + 1):
            candidate = schedule[:pos] + [job] + schedule[pos:]
            ms = _makespan(candidate, p)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        schedule.insert(best_pos, job)
    return schedule


def _two_opt(order, p, max_passes=50):
    """
    2-opt local search: try all pairwise swaps and keep the best improving move.
    """
    n = len(order)
    if n <= 2:
        return order
    
    current = list(order)
    current_ms = _makespan(current, p)
    
    for _ in range(max_passes):
        improved = False
        for i in range(n):
            for j in range(i + 1, n):
                # Swap jobs at positions i and j
                candidate = current[:]
                candidate[i], candidate[j] = candidate[j], candidate[i]
                ms = _makespan(candidate, p)
                if ms < current_ms - 1e-10:
                    current = candidate
                    current_ms = ms
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    
    return current


def order_jobs(processing_times):
    """
    Find a good permutation of jobs to minimize makespan on a flow-shop.
    Uses NEH heuristic with 2-opt local search and multiple restarts.
    """
    p = np.asarray(processing_times, dtype=np.float64)
    n, m = p.shape
    
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s safety margin
    
    # Initial solution using NEH
    best_order = _neh_order(p, None)
    best_ms = _makespan(best_order, p)
    
    # Improve with 2-opt
    best_order = _two_opt(best_order, p, max_passes=100)
    best_ms = _makespan(best_order, p)
    
    elapsed = time.time() - start_time
    remaining = deadline - time.time()
    
    if remaining > 5.0:
        # Try multiple restarts with perturbed NEH
        rng = np.random.default_rng(42)
        restarts = 0
        while time.time() < deadline:
            # Perturb: add small random noise to totals for tie-breaking variation
            totals = p.sum(axis=1)
            noise = rng.uniform(0, 0.1 * np.max(totals), size=n)
            order = list(np.argsort(-(totals + noise), kind='stable'))
            
            # Build schedule by insertion (NEH-style)
            schedule = []
            for job in order:
                best_pos = 0
                best_ms_c = np.inf
                for pos in range(len(schedule) + 1):
                    candidate = schedule[:pos] + [job] + schedule[pos:]
                    ms = _makespan(candidate, p)
                    if ms < best_ms_c:
                        best_ms_c = ms
                        best_pos = pos
                schedule.insert(best_pos, job)
            
            # Improve with 2-opt
            schedule = _two_opt(schedule, p, max_passes=50)
            ms = _makespan(schedule, p)
            
            if ms < best_ms:
                best_ms = ms
                best_order = schedule
            
            restarts += 1
            # Check time
            if time.time() > deadline:
                break
    
    return best_order
# EVOLVE-BLOCK-END
