"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(processing_times, order):
    """Compute makespan for a given permutation."""
    n_jobs, n_machines = processing_times.shape
    completion = np.zeros(n_machines)
    for job in order:
        for m in range(n_machines):
            if m == 0:
                completion[m] += processing_times[job, m]
            else:
                completion[m] = max(completion[m], completion[m-1]) + processing_times[job, m]
    return completion[-1]


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Initial solution: NEH-style heuristic
    # Sort by total processing time descending
    totals = processing_times.sum(axis=1)
    initial_order = list(np.argsort(-totals))
    
    best_order = initial_order.copy()
    best_makespan = _makespan(processing_times, best_order)
    
    # Local search: 2-opt style neighborhood
    # Try swapping pairs of jobs and reversing segments
    n = n_jobs
    
    # Precompute for speed
    pt = processing_times
    
    def compute_makespan_fast(order):
        """Vectorized-ish makespan computation."""
        completion = 0.0
        for m in range(n_machines):
            cumsum = 0.0
            for job in order:
                cumsum += pt[job, m]
            # This isn't quite right for flow shop, need proper simulation
        # Proper simulation
        machine_times = np.zeros(n_machines)
        for job in order:
            for m in range(n_machines):
                if m == 0:
                    machine_times[m] += pt[job, m]
                else:
                    machine_times[m] = max(machine_times[m], machine_times[m-1]) + pt[job, m]
        return machine_times[-1]
    
    # Use a more efficient makespan calculation
    def makespan_calc(order):
        if n_machines == 1:
            return sum(pt[job, 0] for job in order)
        
        # For each machine, track completion time
        # completion[m] = time when machine m finishes current job
        comp = [0.0] * n_machines
        for job in order:
            for m in range(n_machines):
                if m == 0:
                    comp[m] += pt[job, m]
                else:
                    comp[m] = max(comp[m], comp[m-1]) + pt[job, m]
        return comp[-1]
    
    best_makespan = makespan_calc(best_order)
    
    # Local search with swaps
    swapped = True
    while swapped:
        swapped = False
        for i in range(n - 1):
            if time.time() > deadline:
                return best_order
            
            for j in range(i + 1, n):
                if time.time() > deadline:
                    return best_order
                
                # Try swapping i and j
                new_order = best_order.copy()
                new_order[i], new_order[j] = new_order[j], new_order[i]
                new_ms = makespan_calc(new_order)
                
                if new_ms < best_makespan:
                    best_order = new_order
                    best_makespan = new_ms
                    swapped = True
    
    # Also try insert moves
    for i in range(n):
        if time.time() > deadline:
            return best_order
        
        for j in range(n):
            if i == j:
                continue
            if time.time() > deadline:
                return best_order
            
            new_order = best_order.copy()
            # Remove job at i and insert at j
            job = new_order.pop(i)
            if j > i:
                j -= 1
            new_order.insert(j, job)
            
            new_ms = makespan_calc(new_order)
            if new_ms < best_makespan:
                best_order = new_order
                best_makespan = new_ms
    
    return best_order
# EVOLVE-BLOCK-END
