"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(processing_times, order):
    """Compute makespan for a given job order efficiently."""
    n_jobs = len(order)
    if n_jobs == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    
    # c[j] = completion time of machine j after processing the last job in 'order'
    c = [0.0] * n_machines
    
    for i in range(n_jobs):
        job_idx = order[i]
        p_row = processing_times[job_idx]
        for j in range(n_machines):
            if j == 0:
                c[j] += p_row[j]
            else:
                c[j] = max(c[j], c[j-1]) + p_row[j]
    
    return c[-1]

def _makespan_delta_swap(processing_times, order, i, j):
    """Compute the change in makespan if we swap positions i and j in order.
    Returns the delta (new_makespan - old_makespan).
    This is an approximation that recomputes only the affected segment,
    but for correctness in a local search we often just recompute.
    For speed, we can try a quick check, but recomputing the whole makespan
    is O(n*m). With n~100, m~10, it's 1000 ops. 
    Let's stick to full recomputation for simplicity and correctness first,
    then optimize if needed.
    """
    new_order = list(order)
    new_order[i], new_order[j] = new_order[j], new_order[i]
    return _makespan(processing_times, new_order) - _makespan(processing_times, order)

def _makespan_delta_insert(processing_times, order, i, j):
    """Move job at position i to position j. Returns delta."""
    if i == j:
        return 0.0
    new_order = list(order)
    job = new_order.pop(i)
    new_order.insert(j, job)
    return _makespan(processing_times, new_order) - _makespan(processing_times, order)

def _neh_rule(processing_times):
    """
    NEH heuristic for flow shop.
    """
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    # Sort jobs by total processing time in descending order
    job_indices = np.argsort(-totals)
    
    order = []
    for job in job_indices:
        # Try all positions to insert this job
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(order) + 1):
            trial = order[:pos] + [job] + order[pos:]
            ms = _makespan(processing_times, trial)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        order = order[:best_pos] + [job] + order[best_pos:]
    
    return order

def _local_search(processing_times, order, time_limit, rng):
    """
    2-opt and insertion local search.
    """
    n = len(order)
    current_order = list(order)
    current_ms = _makespan(processing_times, current_order)
    
    improved = True
    while improved:
        improved = False
        start_time = time.time()
        
        # Swap moves
        for i in range(n):
            if time.time() - start_time > time_limit * 0.5:
                break
            for j in range(i+1, n):
                if time.time() - start_time > time_limit * 0.5:
                    break
                new_order = current_order[:]
                new_order[i], new_order[j] = new_order[j], new_order[i]
                new_ms = _makespan(processing_times, new_order)
                if new_ms < current_ms:
                    current_order = new_order
                    current_ms = new_ms
                    improved = True
        
        # Insertion moves
        for i in range(n):
            if time.time() - start_time > time_limit * 0.8:
                break
            job = current_order[i]
            for j in range(n):
                if i == j:
                    continue
                if time.time() - start_time > time_limit * 0.8:
                    break
                new_order = current_order[:]
                new_order.pop(i)
                # Adjust j if i < j
                insert_pos = j
                if i < j:
                    insert_pos = j - 1
                new_order.insert(insert_pos, job)
                new_ms = _makespan(processing_times, new_order)
                if new_ms < current_ms:
                    current_order = new_order
                    current_ms = new_ms
                    improved = True
                    break # restart insertion loop for this job? or just continue
    
    return current_order, current_ms

def order_jobs(processing_times):
    # Ensure input is numpy array
    p = np.asarray(processing_times, dtype=float)
    n_jobs, n_machines = p.shape
    
    # Start with NEH
    order = _neh_rule(p)
    ms = _makespan(p, order)
    
    # Set a deadline: 110 seconds to be safe
    start_time = time.time()
    deadline = start_time + 110
    
    # Local search phase
    while time.time() < deadline:
        order, ms = _local_search(p, order, 10.0, None)
        
        # Check if we should stop early if no improvement in a long time?
        # For now, just loop until time runs out.
        
        # Random perturbation to escape local optima
        if time.time() < deadline - 5:
            # Simple random swap
            i = random.randint(0, n_jobs - 1)
            j = random.randint(0, n_jobs - 1)
            if i != j:
                order[i], order[j] = order[j], order[i]
                new_ms = _makespan(p, order)
                # Accept if better or with some probability (SA-like)
                # For simplicity, just accept if better, otherwise revert
                if new_ms > ms:
                    order[i], order[j] = order[j], order[i]
                else:
                    ms = new_ms
    
    return list(order)
# EVOLVE-BLOCK-END
