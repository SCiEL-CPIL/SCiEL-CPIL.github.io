"""Improved program for permutation flow-shop construction using a search portfolio."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation on the given processing times matrix."""
    n_jobs, n_machines = processing_times.shape
    completion = np.zeros((n_jobs, n_machines))
    for j in range(n_jobs):
        job_idx = perm[j]
        for m in range(n_machines):
            if m == 0:
                if j == 0:
                    completion[j, m] = processing_times[job_idx, m]
                else:
                    completion[j, m] = completion[j-1, m] + processing_times[job_idx, m]
            else:
                if j == 0:
                    completion[j, m] = processing_times[job_idx, m]
                else:
                    completion[j, m] = max(completion[j-1, m] + processing_times[job_idx, m],
                                           completion[j, m-1])
    return completion[-1, -1]


def _generate_neh_order(processing_times):
    """Generate NEH order."""
    n_jobs = processing_times.shape[0]
    sums = processing_times.sum(axis=1)
    # Sort by decreasing sum
    order = list(np.argsort(-sums))
    return order


def _generate_random_order(processing_times):
    """Generate random order."""
    n_jobs = processing_times.shape[0]
    return list(np.random.permutation(n_jobs))


def _improve_order(order, processing_times, max_iters=1000):
    """Improve order using 2-opt moves."""
    n_jobs = len(order)
    if n_jobs <= 2:
        return order
    
    current_order = list(order)
    current_makespan = _makespan(current_order, processing_times)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n_jobs - 1):
            for j in range(i + 1, n_jobs):
                # Try swapping jobs at positions i and j
                new_order = current_order.copy()
                new_order[i], new_order[j] = new_order[j], new_order[i]
                new_makespan = _makespan(new_order, processing_times)
                if new_makespan < current_makespan:
                    current_order = new_order
                    current_makespan = new_makespan
                    improved = True
        if not improved:
            break
    
    return current_order


def order_jobs(processing_times):
    """Main function to order jobs."""
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds deadline with 10 second safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    # Generate initial solutions
    candidates = []
    
    # NEH order
    neh_order = _generate_neh_order(processing_times)
    candidates.append(neh_order)
    
    # Random orders
    n_random = min(50, n_jobs * 5)
    for _ in range(n_random):
        random_order = _generate_random_order(processing_times)
        candidates.append(random_order)
    
    # Evaluate all candidates and keep the best
    best_order = candidates[0]
    best_makespan = _makespan(best_order, processing_times)
    
    for order in candidates[1:]:
        makespan = _makespan(order, processing_times)
        if makespan < best_makespan:
            best_order = order
            best_makespan = makespan
    
    # Check time
    if time.time() > deadline:
        return list(best_order)
    
    # Improve the best order
    improved_order = _improve_order(best_order, processing_times, max_iters=500)
    improved_makespan = _makespan(improved_order, processing_times)
    
    if improved_makespan < best_makespan:
        best_order = improved_order
        best_makespan = improved_makespan
    
    # Check time
    if time.time() > deadline:
        return list(best_order)
    
    # Try improving other good candidates
    # Sort candidates by makespan
    candidate_scores = []
    for order in candidates:
        makespan = _makespan(order, processing_times)
        candidate_scores.append((makespan, order))
    
    candidate_scores.sort()
    
    # Try improving top 5 candidates
    for makespan, order in candidate_scores[:5]:
        if time.time() > deadline:
            break
        
        improved = _improve_order(order, processing_times, max_iters=300)
        improved_makespan = _makespan(improved, processing_times)
        
        if improved_makespan < best_makespan:
            best_order = improved
            best_makespan = improved_makespan
    
    return list(best_order)
# EVOLVE-BLOCK-END
