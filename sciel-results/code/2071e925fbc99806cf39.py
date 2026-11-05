"""Improved flow-shop scheduling via NEH heuristic with local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, order):
    """Compute makespan for a given job order on a flow shop."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0.0
    # completion_times[j] = completion time of job j on last machine
    # We track the earliest start time for each job on each machine
    # completion[j][m] = completion time of job j on machine m
    # But we can do it more efficiently:
    # For each machine, the completion time of job j depends on:
    #   - completion time of job j on previous machine
    #   - completion time of previous job on this machine
    
    # Let's use a 1D approach:
    # For a given order, we process jobs in that order.
    # For each job, we compute completion times on each machine sequentially.
    
    # machine_completion[m] = when machine m becomes free (i.e., completion of last job on m)
    machine_completion = np.zeros(n_machines)
    makespan = 0.0
    
    for job_idx in order:
        # For this job, compute completion on each machine
        prev_completion = 0.0  # completion on previous machine for this job
        for m in range(n_machines):
            # Start time is max of when previous machine finished this job, and when this machine is free
            start_time = max(prev_completion, machine_completion[m])
            prev_completion = start_time + processing_times[job_idx, m]
            machine_completion[m] = prev_completion
        makespan = max(makespan, prev_completion)
    
    return makespan


def _neh_heuristic(processing_times):
    """NEH heuristic for flow shop scheduling."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    # Insert each job into the best position in the sequence
    sequence = [sorted_jobs[0]]
    
    for job in sorted_jobs[1:]:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence = sequence[:best_pos] + [job] + sequence[best_pos:]
    
    return sequence


def _palmer(processing_times):
    """Palmer's priority rule: sort by decreasing (n+1-j)*t_j1 - j*t_jn."""
    n_jobs, n_machines = processing_times.shape
    if n_machines == 0:
        return list(range(n_jobs))
    priorities = np.zeros(n_jobs)
    for j in range(n_jobs):
        priorities[j] = (n_jobs + 1 - j) * processing_times[j, 0] - j * processing_times[j, n_machines - 1]
    return list(np.argsort(-priorities))


def _local_search(processing_times, initial_order, time_limit=10.0):
    """2-opt local search with time limit."""
    n_jobs = len(initial_order)
    if n_jobs <= 1:
        return list(initial_order)
    
    current_order = list(initial_order)
    current_makespan = _makespan(processing_times, current_order)
    best_order = list(current_order)
    best_makespan = current_makespan
    
    start_time = time.time()
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        # Try all 2-opt moves: swap two jobs
        for i in range(n_jobs - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n_jobs):
                if time.time() - start_time > time_limit:
                    break
                
                # Try swapping jobs at positions i and j
                new_order = current_order[:]
                new_order[i], new_order[j] = new_order[j], new_order[i]
                new_makespan = _makespan(processing_times, new_order)
                
                if new_makespan < current_makespan - 1e-9:
                    current_order = new_order
                    current_makespan = new_makespan
                    if current_makespan < best_makespan:
                        best_makespan = current_makespan
                        best_order = list(current_order)
                    improved = True
                    break
            if improved:
                break
        
        if not improved:
            break
    
    return best_order


def _random_perturbation(order, strength=0.3, rng=None):
    """Apply random 2-opt perturbation."""
    n = len(order)
    if n <= 1:
        return list(order)
    
    if rng is None:
        rng = np.random
    
    new_order = list(order)
    num_swaps = max(1, int(strength * n))
    for _ in range(num_swaps):
        i, j = rng.choice(n, 2, replace=False)
        new_order[i], new_order[j] = new_order[j], new_order[i]
    
    return new_order


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow shop.
    Uses NEH heuristic followed by local search with time management.
    """
    processing_times = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Deadline guard: we must return well before 150s
    deadline = 140.0  # 10s safety margin
    start_time = time.time()
    
    # Step 1: NEH heuristic
    neh_order = _neh_heuristic(processing_times)
    neh_makespan = _makespan(processing_times, neh_order)
    
    best_order = list(neh_order)
    best_makespan = neh_makespan
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if remaining < 5.0:
        return best_order
    
    # Step 2: Local search on the NEH solution
    ls_time = min(remaining * 0.5, 60.0)  # Use up to half remaining time, max 60s
    ls_order = _local_search(processing_times, best_order, time_limit=ls_time)
    ls_makespan = _makespan(processing_times, ls_order)
    
    if ls_makespan < best_makespan:
        best_order = list(ls_order)
        best_makespan = ls_makespan
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if remaining < 5.0:
        return best_order
    
    # Step 3: Multiple restarts with perturbation
    rng = np.random.RandomState(42)  # Reproducibility
    num_restarts = min(20, int(remaining / 2.0))  # Budget ~2s per restart
    
    for r in range(num_restarts):
        if time.time() + 2.0 > deadline:
            break
        
        # Perturb the best solution
        perturbed = _random_perturbation(best_order, strength=0.2 + 0.1 * r, rng=rng)
        
        # Local search on perturbed solution
        ls_time_r = min(1.5, deadline - time.time() - 1.0)
        if ls_time_r < 0.1:
            break
        
        ls_order_r = _local_search(processing_times, perturbed, time_limit=ls_time_r)
        ls_makespan_r = _makespan(processing_times, ls_order_r)
        
        if ls_makespan_r < best_makespan:
            best_order = list(ls_order_r)
            best_makespan = ls_makespan_r
    
    return best_order
# EVOLVE-BLOCK-END
