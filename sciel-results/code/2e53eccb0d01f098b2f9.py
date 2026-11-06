"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(order, pt_list):
    """
    Compute makespan for a given job order.
    pt_list is a list of lists for faster access.
    """
    n_machines = len(pt_list[0]) if pt_list else 0
    if n_machines == 0 or len(order) == 0:
        return 0.0
    
    completion = [0.0] * n_machines
    
    for job in order:
        p = pt_list[job]
        prev = 0.0
        for m in range(n_machines):
            if prev < completion[m]:
                prev = completion[m]
            prev += p[m]
            completion[m] = prev
    return completion[-1]


def _neh_initialization(processing_times, seed=None):
    """
    NEH heuristic with randomization for diversity.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    if seed is not None:
        rng = random.Random(seed)
    else:
        rng = random.Random()
    
    # Order jobs by decreasing total processing time
    totals = processing_times.sum(axis=1)
    
    # Add randomization to break ties and create diversity
    range_totals = totals.max() - totals.min()
    if range_totals < 1e-9:
        range_totals = 1.0
        
    # Use a small fraction of the range for perturbation
    perturbation = [rng.random() * 0.1 * range_totals for _ in range(n_jobs)]
    perturbed_totals = [totals[i] + perturbation[i] for i in range(n_jobs)]
    
    # Sort jobs by decreasing perturbed total
    job_indices = list(range(n_jobs))
    job_indices.sort(key=lambda j: perturbed_totals[j], reverse=True)
    
    # Build sequence greedily using NEH rule
    seq = [job_indices[0]]
    
    # Convert to list of lists for faster makespan calculation
    pt_list = processing_times.tolist()
    
    for job in job_indices[1:]:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(new_seq, pt_list)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    
    return seq


def _local_search(order, pt_list, time_limit=50.0):
    """
    Local search using insertion moves with First Improvement strategy.
    """
    start_time = time.time()
    n_jobs = len(order)
    if n_jobs <= 1:
        return order, _makespan(order, pt_list)
    
    best_order = list(order)
    best_ms = _makespan(best_order, pt_list)
    
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        for i in range(n_jobs):
            if time.time() - start_time >= time_limit:
                return best_order, best_ms
            
            job = best_order[i]
            
            for j in range(n_jobs):
                if j == i or j == i + 1:
                    continue
                
                if j < i:
                    new_order = best_order[:j] + [job] + best_order[j:i] + best_order[i+1:]
                else:
                    new_order = best_order[:i] + best_order[i+1:j+1] + [job] + best_order[j+1:]
                
                ms = _makespan(new_order, pt_list)
                
                if ms < best_ms - 1e-9:
                    best_order = new_order
                    best_ms = ms
                    improved = True
                    break
            
            if improved:
                pass

    return best_order, best_ms


def order_jobs(processing_times):
    """
    Main function: construct a job permutation minimizing makespan.
    Uses multiple NEH restarts and local search.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    pt_list = processing_times.tolist()
    
    best_order = None
    best_ms = float('inf')
    
    # Determine number of restarts
    # If N is small, we can do more restarts.
    # If N is large, local search is slower, so fewer restarts.
    n_restarts = 20 if n_jobs < 30 else 10
    
    for r in range(n_restarts):
        if time.time() >= deadline - 5.0:
            break
        
        seed = r * 12345 + 99
        initial = _neh_initialization(processing_times, seed=seed)
        
        # Apply local search
        remaining = deadline - time.time() - 2.0
        if remaining > 1.0:
            improved, ms = _local_search(initial, pt_list, time_limit=remaining)
            if ms < best_ms:
                best_ms = ms
                best_order = improved
    
    if best_order is None:
        best_order = _neh_initialization(processing_times)
    
    return best_order
# EVOLVE-BLOCK-END
