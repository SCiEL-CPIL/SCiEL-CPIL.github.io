"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(processing_times, order):
    """Compute makespan for a given job order efficiently."""
    if len(order) == 0:
        return 0.0
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    # c_prev[j] is C[i-1][j]
    # c_curr[j] is C[i][j]
    c_prev = [0.0] * n_machines
    c_curr = [0.0] * n_machines
    
    for i in range(n_jobs):
        job_idx = order[i]
        p_row = processing_times[job_idx]
        for j in range(n_machines):
            p = p_row[j]
            if i == 0 and j == 0:
                c_curr[j] = p
            elif i == 0:
                c_curr[j] = c_curr[j-1] + p
            elif j == 0:
                c_curr[j] = c_prev[j] + p
            else:
                c_curr[j] = max(c_prev[j], c_curr[j-1]) + p
        # Swap for next iteration
        c_prev, c_curr = c_curr, c_prev
    
    return c_prev[-1]

def _neh_rule(processing_times):
    """NEH heuristic for flow shop scheduling."""
    n_jobs, n_machines = processing_times.shape
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    # Insert each job into the best position
    order = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(order) + 1):
            candidate = order[:pos] + [job] + order[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        order.insert(best_pos, job)
    
    return order

def _local_search(processing_times, order, time_limit=10.0):
    """
    Local search with swap, insertion, and 2-opt (reversal) moves.
    Uses first-improvement strategy within a time limit.
    """
    n_jobs = len(order)
    if n_jobs <= 1:
        return list(order), _makespan(processing_times, order)
    
    best_order = list(order)
    best_ms = _makespan(processing_times, best_order)
    start_time = time.time()
    
    # We will iterate until time runs out or no improvement is found in a full pass
    # To speed up, we don't do full passes if time is tight. We just try moves.
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # 1. Try swaps
        for i in range(n_jobs - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n_jobs):
                if time.time() - start_time > time_limit:
                    break
                
                # Perform swap
                best_order[i], best_order[j] = best_order[j], best_order[i]
                ms = _makespan(processing_times, best_order)
                if ms < best_ms:
                    best_ms = ms
                    improved = True
                else:
                    # Revert swap
                    best_order[i], best_order[j] = best_order[j], best_order[i]
        
        if time.time() - start_time > time_limit:
            break
            
        # 2. Try insertion moves (remove job i, insert at position j)
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            job_to_move = best_order[i]
            for j in range(n_jobs):
                if i == j or (i == j - 1):
                    continue
                if time.time() - start_time > time_limit:
                    break
                
                # Perform insertion
                # Remove job from i
                candidate = best_order[:i] + best_order[i+1:]
                # Adjust j if j > i because the list shifted
                insert_pos = j if j > i else j
                candidate.insert(insert_pos, job_to_move)
                
                ms = _makespan(processing_times, candidate)
                if ms < best_ms:
                    best_order = candidate
                    best_ms = ms
                    improved = True
        
        if time.time() - start_time > time_limit:
            break
            
        # 3. Try 2-opt (reversal)
        for i in range(n_jobs - 1):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n_jobs):
                if time.time() - start_time > time_limit:
                    break
                
                # Reverse segment i to j
                best_order[i:j+1] = best_order[i:j+1][::-1]
                ms = _makespan(processing_times, best_order)
                if ms < best_ms:
                    best_ms = ms
                    improved = True
                else:
                    # Revert reverse
                    best_order[i:j+1] = best_order[i:j+1][::-1]
        
        if not improved:
            break
            
    return best_order, best_ms

def _perturb(order, strength=1):
    """
    Perturb a solution by performing 'strength' random swaps.
    """
    perturbed = list(order)
    n = len(perturbed)
    for _ in range(strength):
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        if i != j:
            perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
    return perturbed

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    
    Uses NEH heuristic followed by time-limited local search with restarts.
    """
    start_time = time.time()
    # Safety margin: The evaluator kills at 120s. Use 110s to be safe.
    deadline = start_time + 110.0
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Generate initial solution using NEH
    initial_order = _neh_rule(processing_times)
    initial_ms = _makespan(processing_times, initial_order)
    
    # Run local search with time budget
    remaining_time = deadline - time.time()
    if remaining_time < 5.0:
        return initial_order
    
    # Allocate most time to the first local search
    # Reserve 10s for potential restarts if the first search finishes early
    time_limit = remaining_time - 10.0
    if time_limit < 1.0:
        time_limit = remaining_time - 1.0
        
    best_order, best_ms = _local_search(
        processing_times, initial_order, 
        time_limit=time_limit
    )
    
    # If we have significant time left, try restarts
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        best_overall = best_order
        best_overall_ms = best_ms
        
        # Try a few random restarts with short time limits
        # We will try to fit as many restarts as possible
        while time.time() < deadline - 5.0:
            # Perturb the best solution found so far
            # Strength increases slightly or stays constant
            perturbed_order = _perturb(best_overall, strength=2)
            
            # Quick local search with shorter time limit
            tl = min(remaining_time - 5.0, 5.0)
            if tl <= 0:
                break
            
            order, ms = _local_search(
                processing_times, perturbed_order,
                time_limit=tl
            )
            
            if ms < best_overall_ms:
                best_overall = order
                best_overall_ms = ms
            
            remaining_time = deadline - time.time()
            if remaining_time <= 5.0:
                break
        
        return best_overall
    
    return best_order
# EVOLVE-BLOCK-END
