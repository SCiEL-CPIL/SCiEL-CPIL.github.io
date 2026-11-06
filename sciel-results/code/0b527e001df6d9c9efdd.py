"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation efficiently."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    
    # completion[m] tracks the completion time on machine m after processing all jobs so far
    # For job j in permutation, on machine m:
    #   completion[m] = max(completion[m], completion[m-1]) + processing_times[perm[j]][m]
    # We maintain an array of machine completion times.
    
    machine_times = np.zeros(n_machines, dtype=np.float64)
    for j in range(n_jobs):
        job = perm[j]
        pt = processing_times[job]
        # Process each machine sequentially
        # machine_times[m] = max(machine_times[m], machine_times[m-1]) + pt[m]
        # We need to do this in-place carefully
        prev = 0.0
        for m in range(n_machines):
            new_time = max(machine_times[m], prev) + pt[m]
            machine_times[m] = new_time
            prev = new_time
    
    return machine_times[-1]


def _neh_order(processing_times):
    """NEH (Nawaz-Enscore-Ham) heuristic for flow shop scheduling."""
    n_jobs, n_machines = processing_times.shape
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    # Insert jobs one by one into the best position
    perm = []
    for job in sorted_jobs:
        # Try inserting at each position
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm


def _neighborhood_search(perm, processing_times, time_limit=5.0):
    """Improve permutation using 2-opt style neighborhood search."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm, _makespan(perm, processing_times)
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    start_time = time.time()
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        for i in range(n_jobs - 1):
            if (time.time() - start_time) >= time_limit:
                break
            for j in range(i + 1, n_jobs):
                # Swap jobs at positions i and j
                candidate = list(best_perm)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                ms = _makespan(candidate, processing_times)
                if ms < best_ms - 1e-9:
                    best_ms = ms
                    best_perm = candidate
                    improved = True
    
    return best_perm, best_ms


def _simulated_annealing(perm, processing_times, time_limit=10.0):
    """Simulated annealing for further improvement."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm, _makespan(perm, processing_times)
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    current_perm = list(best_perm)
    current_ms = best_ms
    
    start_time = time.time()
    t0 = 100.0
    t_end = 1.0
    cooling_rate = 0.995
    
    temperature = t0
    while temperature > t_end and (time.time() - start_time) < time_limit:
        # Generate a random move: swap two random positions
        i = random.randint(0, n_jobs - 1)
        j = random.randint(0, n_jobs - 1)
        if i == j:
            continue
        
        candidate = list(current_perm)
        candidate[i], candidate[j] = candidate[j], candidate[i]
        candidate_ms = _makespan(candidate, processing_times)
        
        delta = candidate_ms - current_ms
        if delta < 0 or random.random() < np.exp(-delta / temperature):
            current_perm = candidate
            current_ms = candidate_ms
            if current_ms < best_ms - 1e-9:
                best_ms = current_ms
                best_perm = list(current_perm)
        
        temperature *= cooling_rate
    
    return best_perm, best_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    Uses NEH heuristic followed by neighborhood search and simulated annealing.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Set a global deadline with safety margin (120s limit, use 110s)
    global_deadline = time.time() + 110.0
    
    # Step 1: NEH heuristic
    neh_perm = _neh_order(processing_times)
    neh_ms = _makespan(neh_perm, processing_times)
    
    # Step 2: Neighborhood search (2-opt swaps)
    ns_time_limit = min(10.0, max(1.0, (global_deadline - time.time()) * 0.3))
    ns_perm, ns_ms = _neighborhood_search(neh_perm, processing_times, time_limit=ns_time_limit)
    
    # Step 3: Simulated annealing if time remains
    remaining_time = global_deadline - time.time()
    if remaining_time > 2.0:
        sa_time_limit = min(remaining_time - 1.0, 30.0)
        sa_perm, sa_ms = _simulated_annealing(ns_perm, processing_times, time_limit=sa_time_limit)
        if sa_ms < ns_ms:
            return sa_perm
    
    return ns_perm
# EVOLVE-BLOCK-END
