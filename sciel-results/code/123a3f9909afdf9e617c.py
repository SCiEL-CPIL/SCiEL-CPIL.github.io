"""Improved program for permutation flow-shop construction using NEH + local search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(perm, pt):
    """Compute makespan for a given permutation.
    
    Args:
        perm: list/array of job indices in order
        pt: 2D array of shape (n_jobs, n_machines)
    
    Returns:
        makespan (float)
    """
    n_jobs = len(perm)
    if n_jobs == 0:
        return 0.0
    n_machines = pt.shape[1]
    
    # C[j, m] = completion time of job j on machine m
    # We can compute this incrementally
    # For flow shop: C(j, m) = max(C(j-1, m), C(j, m-1)) + pt[perm[j], m]
    # where C(-1, m) = 0 and C(j, -1) = 0
    
    # Use a 1D array for the previous row
    prev_row = np.zeros(n_machines)  # C(j-1, m)
    current_cum = 0.0  # C(j, m-1) accumulated
    
    for j in range(n_jobs):
        job = perm[j]
        for m in range(n_machines):
            current_cum = max(prev_row[m], current_cum) + pt[j, m]
            prev_row[m] = current_cum
    
    return prev_row[-1]


def _makespan_delta(perm, pt, i, j):
    """Compute the change in makespan when swapping positions i and j in perm.
    
    This is an approximation; for exact delta we'd need to recompute the affected segment.
    For simplicity, we'll just recompute the full makespan when needed.
    """
    pass


def _generate_neh_order(pt):
    """Generate an initial ordering using the NEH heuristic."""
    n_jobs, n_machines = pt.shape
    
    # Sum processing times for each job
    totals = pt.sum(axis=1)
    
    # Sort jobs in descending order of total processing time
    sorted_jobs = list(np.argsort(-totals))
    
    # Build the permutation incrementally using NEH
    perm = []
    remaining = set(sorted_jobs)
    
    for job in sorted_jobs:
        remaining.remove(job)
        
        if not perm:
            perm.append(job)
            continue
        
        # Find the best position to insert this job
        best_pos = 0
        best_ms = float('inf')
        
        # Try all possible insertion positions
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(candidate, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        perm.insert(best_pos, job)
    
    return perm


def _generate_random_order(n_jobs):
    """Generate a random permutation."""
    perm = list(range(n_jobs))
    random.shuffle(perm)
    return perm


def _swap_local_search(perm, pt, time_limit):
    """Perform swap-based local search until time limit or no improvement."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, pt)
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 5000  # Stop if no improvement for this many iterations
    
    while no_improve_count < max_no_improve:
        if time.time() - start_time > time_limit:
            break
        
        # Try random swaps
        improved = False
        i = random.randint(0, n_jobs - 1)
        j = random.randint(0, n_jobs - 1)
        if i == j:
            no_improve_count += 1
            continue
        
        # Swap
        best_perm[i], best_perm[j] = best_perm[j], best_perm[i]
        ms = _makespan(best_perm, pt)
        
        if ms < best_ms:
            best_ms = ms
            no_improve_count = 0
            improved = True
        else:
            # Undo swap
            best_perm[i], best_perm[j] = best_perm[j], best_perm[i]
            no_improve_count += 1
    
    return best_perm


def _adjacent_local_search(perm, pt, time_limit):
    """Perform adjacent swap local search."""
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, pt)
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 10000
    
    while no_improve_count < max_no_improve:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        # Try random adjacent swaps
        i = random.randint(0, n_jobs - 2)
        j = i + 1
        
        # Swap
        best_perm[i], best_perm[j] = best_perm[j], best_perm[i]
        ms = _makespan(best_perm, pt)
        
        if ms < best_ms:
            best_ms = ms
            no_improve_count = 0
            improved = True
        else:
            # Undo swap
            best_perm[i], best_perm[j] = best_perm[j], best_perm[i]
            no_improve_count += 1
    
    return best_perm


def _multi_start_local_search(pt, time_budget):
    """Run multiple starts with local search."""
    n_jobs, n_machines = pt.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Generate initial solutions
    neh_perm = _generate_neh_order(pt)
    
    # Track best solution
    best_perm = list(neh_perm)
    best_ms = _makespan(best_perm, pt)
    
    start_time = time.time()
    
    # Do local search on NEH solution first
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 0.1:
        ls_perm = _swap_local_search(best_perm, pt, remaining_time * 0.3)
        ls_ms = _makespan(ls_perm, pt)
        if ls_ms < best_ms:
            best_ms = ls_ms
            best_perm = ls_perm
    
    # Try additional random starts if time permits
    elapsed = time.time() - start_time
    remaining = time_budget - elapsed
    
    num_random_starts = min(20, max(1, int(remaining / 0.5)))
    
    for _ in range(num_random_starts):
        if time.time() - start_time > time_budget - 0.1:
            break
        
        rand_perm = _generate_random_order(n_jobs)
        rand_ms = _makespan(rand_perm, pt)
        
        if rand_ms < best_ms:
            best_ms = rand_ms
            best_perm = list(rand_perm)
        
        # Do local search on this solution if it's competitive
        if rand_ms < best_ms * 1.1:
            time_left = time_budget - (time.time() - start_time)
            if time_left > 0.1:
                ls_perm = _swap_local_search(best_perm, pt, min(time_left * 0.2, 5.0))
                ls_ms = _makespan(ls_perm, pt)
                if ls_ms < best_ms:
                    best_ms = ls_ms
                    best_perm = ls_perm
    
    return best_perm


def order_jobs(processing_times):
    """Main entry point: order jobs to minimize makespan on a flow shop.
    
    Args:
        processing_times: np.ndarray of shape (n_jobs, n_machines)
    
    Returns:
        List of job indices in the optimal (or near-optimal) order
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Set a deadline: use up to 110 seconds out of the 120s limit
    deadline = 110.0
    
    # For small instances, we can be more thorough
    if n_jobs <= 8:
        # Try all permutations for very small instances
        from itertools import permutations
        best_perm = None
        best_ms = float('inf')
        start_time = time.time()
        
        for perm in permutations(range(n_jobs)):
            if time.time() - start_time > deadline - 5:
                break
            ms = _makespan(perm, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_perm = list(perm)
        
        if best_perm is not None:
            return best_perm
        else:
            return _generate_neh_order(processing_times)
    
    # For larger instances, use multi-start local search
    time_budget = min(deadline, 100.0)
    best_perm = _multi_start_local_search(processing_times, time_budget)
    
    return best_perm
# EVOLVE-BLOCK-END
