"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, order):
    """Compute makespan for a given job order."""
    n_jobs, n_machines = processing_times.shape
    completion = np.zeros(n_machines)
    for job in order:
        for m in range(n_machines):
            completion[m] = max(completion[m], completion[m - 1] if m > 0 else 0) + processing_times[job, m]
    return completion[-1]


def _neh_init(processing_times):
    """NEH heuristic initialization."""
    n_jobs, n_machines = processing_times.shape
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    # Insert each job in the best position
    order = [sorted_jobs[0]]
    for job in sorted_jobs[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(order) + 1):
            new_order = order[:pos] + [job] + order[pos:]
            ms = _makespan(processing_times, new_order)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        order = order[:best_pos] + [job] + order[best_pos:]
    
    return order


def _greedy_insertion_init(processing_times):
    """Greedy insertion construction."""
    n_jobs, n_machines = processing_times.shape
    # Start with the job with longest total processing time
    totals = processing_times.sum(axis=1)
    start_job = int(np.argmax(totals))
    order = [start_job]
    remaining = [j for j in range(n_jobs) if j != start_job]
    
    while remaining:
        best_job = None
        best_pos = 0
        best_ms = float('inf')
        for job in remaining:
            for pos in range(len(order) + 1):
                new_order = order[:pos] + [job] + order[pos:]
                ms = _makespan(processing_times, new_order)
                if ms < best_ms:
                    best_ms = ms
                    best_job = job
                    best_pos = pos
        order = order[:best_pos] + [best_job] + order[best_pos:]
        remaining.remove(best_job)
    
    return order


def _local_search(processing_times, order, deadline):
    """Local search with adjacent swaps and reinsertion."""
    n_jobs = len(order)
    current_order = list(order)
    current_ms = _makespan(processing_times, current_order)
    best_order = list(current_order)
    best_ms = current_ms
    
    # Adjacent swap moves
    improved = True
    while improved:
        improved = False
        for i in range(n_jobs - 1):
            if time.time() > deadline:
                break
            new_order = current_order.copy()
            new_order[i], new_order[i + 1] = new_order[i + 1], new_order[i]
            ms = _makespan(processing_times, new_order)
            if ms < current_ms - 1e-9:
                current_order = new_order
                current_ms = ms
                if ms < best_ms:
                    best_ms = ms
                    best_order = list(current_order)
                improved = True
        if time.time() > deadline:
            break
    
    # Reinsertion moves
    improved = True
    while improved:
        improved = False
        for i in range(n_jobs):
            if time.time() > deadline:
                break
            job = current_order[i]
            base = current_order[:i] + current_order[i + 1:]
            for pos in range(len(base) + 1):
                if pos == i:
                    continue
                new_order = base[:pos] + [job] + base[pos:]
                ms = _makespan(processing_times, new_order)
                if ms < current_ms - 1e-9:
                    current_order = new_order
                    current_ms = ms
                    if ms < best_ms:
                        best_ms = ms
                        best_order = list(current_order)
                    improved = True
                    break
        if time.time() > deadline:
            break
    
    return best_order, best_ms


def order_jobs(processing_times):
    """Main entry point: construct a job permutation minimizing makespan."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    processing_times = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = processing_times.shape
    
    # Generate multiple initial solutions
    candidates = []
    
    # NEH initialization
    neh_order = _neh_init(processing_times)
    candidates.append(neh_order)
    
    # Greedy insertion initialization
    gi_order = _greedy_insertion_init(processing_times)
    candidates.append(gi_order)
    
    # Simple LPT as fallback
    lpt_order = list(np.argsort(-processing_times.sum(axis=1)))
    candidates.append(lpt_order)
    
    # Run local search on each candidate within time budget
    best_order = None
    best_ms = float('inf')
    
    for order in candidates:
        if time.time() > deadline:
            break
        ls_order, ls_ms = _local_search(processing_times, order, deadline)
        if ls_ms < best_ms:
            best_ms = ls_ms
            best_order = ls_order
    
    # If no local search was performed (shouldn't happen), return best candidate
    if best_order is None:
        for order in candidates:
            ms = _makespan(processing_times, order)
            if ms < best_ms:
                best_ms = ms
                best_order = order
    
    return best_order
# EVOLVE-BLOCK-END
