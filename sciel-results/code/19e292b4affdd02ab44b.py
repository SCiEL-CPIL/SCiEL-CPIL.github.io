"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, order):
    """Compute makespan for a given job order using flow-shop recurrence.
    
    C[j][m] = max(C[j-1][m], C[j][m-1]) + P[order[j]][m]
    where C[-1][m] = 0 and C[j][-1] = 0.
    """
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    if n_machines == 0:
        return 0.0
    
    # P is the processing time matrix indexed by job
    # We process jobs in the given order
    # completion[j][m] = completion time of job at position j on machine m
    
    # Use a 2D array: rows = positions in order, cols = machines
    completion = np.zeros((n_jobs, n_machines))
    
    for j in range(n_jobs):
        job_idx = order[j]
        for m in range(n_machines):
            p = processing_times[job_idx, m]
            if m == 0:
                if j == 0:
                    completion[j, m] = p
                else:
                    completion[j, m] = completion[j-1, m] + p
            else:
                if j == 0:
                    completion[j, m] = completion[j, m-1] + p
                else:
                    completion[j, m] = max(completion[j-1, m], completion[j, m-1]) + p
    
    return completion[-1, -1]


def _makespan_fast(processing_times, order):
    """Faster makespan using numpy where possible."""
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    if n_jobs == 0 or n_machines == 0:
        return 0.0
    
    # Extract processing times in order
    P = processing_times[order]  # shape (n_jobs, n_machines)
    
    # completion[j][m] = max(completion[j-1][m], completion[j][m-1]) + P[j][m]
    # We can compute this row by row
    completion = np.zeros((n_jobs, n_machines))
    
    for j in range(n_jobs):
        row = P[j]
        if j == 0:
            # First job: completion[0][m] = completion[0][m-1] + P[0][m]
            # This is just cumulative sum along machines
            completion[0] = np.cumsum(row)
        else:
            prev_row = completion[j-1]
            for m in range(n_machines):
                if m == 0:
                    completion[j, m] = prev_row[0] + row[0]
                else:
                    completion[j, m] = max(prev_row[m], completion[j, m-1]) + row[m]
    
    return completion[-1, -1]


def _neh_construction(processing_times):
    """NEH (Nawaz-Enscore-Ham) heuristic for flow shop.
    
    Sort jobs by decreasing total processing time, then insert each job
    at the position that gives minimum partial makespan.
    """
    n_jobs, n_machines = processing_times.shape
    
    # Sort by decreasing total processing time
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    order = []
    for job_idx in sorted_jobs:
        best_pos = len(order)
        best_makespan = np.inf
        for pos in range(len(order) + 1):
            new_order = list(order[:pos]) + [job_idx] + list(order[pos:])
            ms = _makespan_fast(processing_times, new_order)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        order.insert(best_pos, job_idx)
    
    return order


def _insertion_local_search(processing_times, order, max_iterations=10000, time_limit=None):
    """Local search using insertion neighborhood.
    
    For each job, try moving it to every other position.
    Accept the best improving move. Repeat until no improvement or limit reached.
    """
    n_jobs = len(order)
    current_order = list(order)
    current_ms = _makespan_fast(processing_times, current_order)
    
    if time_limit is not None:
        start_time = time.time()
    
    iterations = 0
    improved = True
    while improved:
        improved = False
        best_move = None
        best_new_ms = current_ms
        best_new_order = None
        
        for i in range(n_jobs):
            job = current_order[i]
            # Remove job from position i
            remaining = [x for x in current_order if x != job]
            # Try inserting at each position
            for j in range(n_jobs - 1):
                if j == i or j == i - 1:
                    continue  # Same or adjacent position, no change
                new_order = remaining[:j] + [job] + remaining[j:]
                new_ms = _makespan_fast(processing_times, new_order)
                if new_ms < best_new_ms:
                    best_new_ms = new_ms
                    best_move = (i, j)
                    best_new_order = new_order
        
        if best_new_order is not None:
            current_order = best_new_order
            current_ms = best_new_ms
            improved = True
            iterations += 1
        
        if time_limit is not None:
            if time.time() - start_time > time_limit:
                break
        
        if iterations >= max_iterations:
            break
    
    return current_order, current_ms


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    start_time = time.time()
    time_budget = 120.0  # Safety margin below 150s limit
    safety_margin = 10.0
    
    n_jobs, n_machines = processing_times.shape
    
    # Handle edge cases
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Phase 1: NEH construction
    order = _neh_construction(processing_times)
    best_ms = _makespan_fast(processing_times, order)
    best_order = list(order)
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed > time_budget - safety_margin:
        return best_order
    
    # Phase 2: Local search with insertion neighborhood
    remaining_time = time_budget - safety_margin - elapsed
    if remaining_time > 0.5:
        # Run local search with time limit
        order_ls, ms_ls = _insertion_local_search(
            processing_times, best_order,
            max_iterations=50000,
            time_limit=remaining_time
        )
        if ms_ls < best_ms:
            best_order = order_ls
            best_ms = ms_ls
    
    # Check time again
    elapsed = time.time() - start_time
    if elapsed > time_budget - safety_margin:
        return best_order
    
    # Phase 3: Multiple restarts if time remains
    remaining_time = time_budget - safety_margin - elapsed
    if remaining_time > 2.0 and n_jobs <= 50:
        # Try perturbation + local search for diversification
        n_restarts = min(int(remaining_time / 2.0), 10)
        for r in range(n_restarts):
            if time.time() - start_time > time_budget - safety_margin:
                break
            
            # Perturb the best solution
            perturbed = list(best_order)
            # Random swap of 2-3 pairs
            n_swaps = min(3, n_jobs // 2)
            for _ in range(n_swaps):
                i = np.random.randint(0, n_jobs)
                j = np.random.randint(0, n_jobs)
                if i != j:
                    perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            
            # Local search on perturbed solution
            time_left = time_budget - safety_margin - (time.time() - start_time)
            if time_left < 0.5:
                break
            order_ps, ms_ps = _insertion_local_search(
                processing_times, perturbed,
                max_iterations=10000,
                time_limit=time_left
            )
            if ms_ps < best_ms:
                best_order = order_ps
                best_ms = ms_ps
    
    return best_order
# EVOLVE-BLOCK-END
