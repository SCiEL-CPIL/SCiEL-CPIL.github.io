"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(processing_times, order):
    """Compute makespan for a given job order."""
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    # completion[m] = completion time of last job on machine m
    completion = np.zeros(n_machines, dtype=np.float64)
    for job in order:
        t = 0.0
        for m in range(n_machines):
            t = max(t, completion[m]) + processing_times[job, m]
            completion[m] = t
    return completion[-1]

def _makespan_fast(processing_times, order):
    """Faster makespan computation using cumulative approach."""
    n_machines = processing_times.shape[1]
    if len(order) == 0:
        return 0.0
    # For each machine, track the cumulative time
    # completion on machine m after processing jobs in order
    # C_m(j) = max(C_m(j-1), C_{m-1}(j)) + P[j][m]
    # We can compute this row by row
    n_jobs = len(order)
    # prev_row[m] = C_m for previous job
    prev_row = np.zeros(n_machines, dtype=np.float64)
    for job in order:
        curr_row = np.zeros(n_machines, dtype=np.float64)
        p = processing_times[job]
        for m in range(n_machines):
            if m == 0:
                curr_row[m] = prev_row[m] + p[m]
            else:
                curr_row[m] = max(prev_row[m], curr_row[m-1]) + p[m]
        prev_row = curr_row
    return prev_row[-1]

def _neh_heuristic(processing_times):
    """NEH heuristic for flow-shop scheduling."""
    n_jobs, n_machines = processing_times.shape
    
    # Sort jobs by total processing time in descending order
    total_times = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-total_times))
    
    # Insert each job in the best position
    order = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(order) + 1):
            candidate = order[:pos] + [job] + order[pos:]
            ms = _makespan_fast(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        order.insert(best_pos, job)
    
    return order

def _or_opt_local_search(processing_times, order, max_iter=1000, time_limit=None, start_time=None):
    """Local search using Or-opt (block) moves."""
    if start_time is None:
        start_time = time.time()
    
    n_jobs = len(order)
    if n_jobs <= 2:
        return order
    
    best_order = list(order)
    best_ms = _makespan_fast(processing_times, best_order)
    
    # Or-opt: move a block of 1-3 consecutive jobs to another position
    block_sizes = [1, 2, 3]
    no_improve = 0
    
    for iteration in range(max_iter):
        if time_limit is not None and (time.time() - start_time) > time_limit:
            break
        
        improved = False
        random.shuffle(block_sizes)
        
        for bs in block_sizes:
            if n_jobs < bs:
                continue
            # Try all possible blocks and positions
            for start in range(n_jobs - bs + 1):
                if time_limit is not None and (time.time() - start_time) > time_limit:
                    break
                block = best_order[start:start+bs]
                remaining = best_order[:start] + best_order[start+bs:]
                for pos in range(len(remaining) + 1):
                    if pos == start or (pos == start + bs and bs > 0):
                        # Skip inserting back at same or adjacent position (no change)
                        # Actually need to be more careful
                        if pos <= start and pos >= start + bs:
                            continue
                        # More precise: skip if it results in same order
                        # If we remove from [start, start+bs) and insert at pos,
                        # it's same if pos == start or pos == start+bs (when inserting after the removed block)
                        # Let's just compute and compare
                        candidate = remaining[:pos] + block + remaining[pos:]
                        ms = _makespan_fast(processing_times, candidate)
                        if ms < best_ms - 1e-10:
                            best_order = candidate
                            best_ms = ms
                            improved = True
                            break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= 3:
                break
    
    return best_order

def _aco_construction(processing_times, pheromone, alpha=1.0, beta=2.0, n_jobs=None):
    """Construct a permutation using ACO probabilistic rule."""
    if n_jobs is None:
        n_jobs = processing_times.shape[0]
    
    remaining = list(range(n_jobs))
    order = []
    
    # Compute heuristic values (inverse of processing time, or some measure)
    # Use sum of processing times as heuristic (lower is better for ACO, so we use 1/sum)
    total_times = processing_times.sum(axis=1)
    # Avoid division by zero
    total_times_safe = np.where(total_times == 0, 1e-10, total_times)
    heuristic = 1.0 / total_times_safe
    
    for _ in range(n_jobs):
        if len(remaining) == 1:
            order.append(remaining[0])
            break
        
        # Compute probabilities for each remaining job
        probs = []
        total_prob = 0.0
        for job in remaining:
            # Pheromone from any already-placed job to this job
            # Use average pheromone from placed jobs to this job
            if order:
                pheromone_val = np.mean([pheromone[prev, job] for prev in order])
            else:
                pheromone_val = pheromone[job, job] if n_jobs > 1 else 1.0
            
            prob = (pheromone_val ** alpha) * (heuristic[job] ** beta)
            probs.append(prob)
            total_prob += prob
        
        # Normalize
        if total_prob <= 0:
            # Uniform random
            idx = random.randint(0, len(remaining) - 1)
        else:
            cumsum = np.cumsum(probs)
            cumsum = cumsum / cumsum[-1]
            r = random.random()
            idx = np.searchsorted(cumsum, r)
            if idx >= len(remaining):
                idx = len(remaining) - 1
        
        chosen = remaining[idx]
        order.append(chosen)
        remaining.remove(chosen)
    
    return order

def _update_pheromone(pheromone, order, makespan, evaporation=0.5, rho=0.1, Q=10.0):
    """Update pheromone trails."""
    n_jobs = pheromone.shape[0]
    # Evaporation
    pheromone *= (1.0 - rho)
    
    # Deposit pheromone on the solution path
    if makespan > 0:
        deposit = Q / makespan
        for i in range(n_jobs - 1):
            job_i = order[i]
            job_next = order[i + 1]
            pheromone[job_i, job_next] += deposit
            pheromone[job_next, job_i] += deposit * 0.5  # Symmetric-ish with less weight
    
    # Also add some pheromone for self-loops to encourage starting with good jobs
    total_times = None  # We don't have this here, skip

def _solve(processing_times):
    """Main solver combining ACO with local search."""
    start_time = time.time()
    deadline = 110.0  # Safety margin from 120s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Initialize with NEH heuristic
    best_order = _neh_heuristic(processing_times)
    best_ms = _makespan_fast(processing_times, best_order)
    
    # Apply local search to initial solution
    best_order = _or_opt_local_search(
        processing_times, best_order, 
        max_iter=500, 
        time_limit=10.0, 
        start_time=start_time
    )
    best_ms = _makespan_fast(processing_times, best_order)
    
    # Check if we have time for ACO
    elapsed = time.time() - start_time
    if elapsed > deadline - 30:
        return best_order
    
    # Initialize pheromone
    pheromone = np.ones((n_jobs, n_jobs), dtype=np.float64)
    # Set diagonal to a moderate value
    np.fill_diagonal(pheromone, 0.5)
    
    # ACO parameters
    alpha = 1.0
    beta = 2.0
    rho = 0.1
    Q = 10.0
    n_iterations = 50
    
    aco_start = time.time()
    
    for it in range(n_iterations):
        # Check time
        if time.time() - start_time > deadline - 5:
            break
        
        # Construct solution
        order = _aco_construction(processing_times, pheromone, alpha=alpha, beta=beta, n_jobs=n_jobs)
        
        # Local search improvement
        remaining_time = deadline - 5 - (time.time() - start_time)
        if remaining_time < 1.0:
            break
        
        order = _or_opt_local_search(
            processing_times, order,
            max_iter=200,
            time_limit=min(remaining_time - 0.5, 5.0),
            start_time=start_time
        )
        
        ms = _makespan_fast(processing_times, order)
        
        if ms < best_ms:
            best_ms = ms
            best_order = list(order)
        
        # Update pheromone
        _update_pheromone(pheromone, order, ms, evaporation=0.5, rho=rho, Q=Q)
        
        # Also update with best solution found so far
        _update_pheromone(pheromone, best_order, best_ms, evaporation=0.5, rho=rho * 0.5, Q=Q * 0.5)
        
        # Adaptive parameters
        if it % 10 == 0 and it > 0:
            # Slightly increase exploration over time
            alpha *= 1.05
            beta *= 0.98
    
    return best_order


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    
    Uses a hybrid approach:
    1. NEH heuristic for initial solution
    2. Or-opt local search for improvement
    3. Ant Colony Optimization for global search with local search intensification
    
    Args:
        processing_times: np.ndarray of shape (n_jobs, n_machines)
    
    Returns:
        List of job indices (permutation)
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin
    
    try:
        result = _solve(processing_times)
    except Exception:
        # Fallback to simple heuristic
        result = list(np.argsort(-processing_times.sum(axis=1)))
    
    # Final time check
    if time.time() - start_time > deadline:
        # Should not happen, but just in case
        pass
    
    return result
# EVOLVE-BLOCK-END
