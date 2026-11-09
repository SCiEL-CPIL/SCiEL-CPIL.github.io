"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(processing_times, order):
    """Compute makespan for a given job order efficiently."""
    if len(order) == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    c = np.zeros(n_machines)
    pt = processing_times
    for job_idx in order:
        p_row = pt[job_idx]
        prev_c = 0.0
        for j in range(n_machines):
            c_val = c[j]
            if prev_c > c_val:
                c_val = prev_c
            c_val += p_row[j]
            c[j] = c_val
            prev_c = c_val
    return c[-1]

def _completion_vector(processing_times, order):
    """Compute the completion time vector c for a given job order."""
    if len(order) == 0:
        return np.zeros(processing_times.shape[1])
    n_machines = processing_times.shape[1]
    c = np.zeros(n_machines)
    pt = processing_times
    for job_idx in order:
        p_row = pt[job_idx]
        prev_c = 0.0
        for j in range(n_machines):
            c_val = c[j]
            if prev_c > c_val:
                c_val = prev_c
            c_val += p_row[j]
            c[j] = c_val
            prev_c = c_val
    return c

def _insertion_delta(processing_times, order, c, job, pos):
    """
    Compute the delta in makespan when inserting job at position pos in order.
    Uses the exact recurrence to evaluate the critical path change.
    Returns the new makespan (not delta, for simplicity and accuracy).
    """
    n_machines = processing_times.shape[1]
    p_row = processing_times[job]
    
    # Simulate from position pos onwards
    # We need to compute completion times for jobs from pos to end
    # with the new job inserted at pos
    
    # The completion vector before pos is unchanged
    # We need to propagate from pos onwards
    
    # Get the completion vector state at position pos
    # This is the completion vector after processing all jobs before pos
    c_before = _completion_vector(processing_times, order[:pos])
    
    # Now process: new job at pos, then remaining jobs from pos onwards
    c_new = c_before.copy()
    prev_c = 0.0
    for j in range(n_machines):
        c_val = c_new[j]
        if prev_c > c_val:
            c_val = prev_c
        c_val += p_row[j]
        c_new[j] = c_val
        prev_c = c_val
    
    # Continue with the rest of the jobs
    for job_idx in order[pos:]:
        p_row2 = processing_times[job_idx]
        prev_c = 0.0
        for j in range(n_machines):
            c_val = c_new[j]
            if prev_c > c_val:
                c_val = prev_c
            c_val += p_row2[j]
            c_new[j] = c_val
            prev_c = c_val
    
    return c_new[-1]

def _compute_c_prefix(processing_times, order):
    """Compute prefix completion vectors for all positions in order."""
    n_machines = processing_times.shape[1]
    n = len(order)
    # c_prefix[i] = completion vector after processing order[0:i]
    c_prefix = [np.zeros(n_machines) for _ in range(n + 1)]
    for i in range(n):
        c_prefix[i + 1] = c_prefix[i].copy()
        p_row = processing_times[order[i]]
        prev_c = 0.0
        for j in range(n_machines):
            c_val = c_prefix[i + 1][j]
            if prev_c > c_val:
                c_val = prev_c
            c_val += p_row[j]
            c_prefix[i + 1][j] = c_val
            prev_c = c_val
    return c_prefix

def _makespan_from_suffix(processing_times, order, start_pos, c_start):
    """Compute makespan by processing order[start_pos:] starting from c_start."""
    n_machines = processing_times.shape[1]
    c = c_start.copy()
    pt = processing_times
    for i in range(start_pos, len(order)):
        job_idx = order[i]
        p_row = pt[job_idx]
        prev_c = 0.0
        for j in range(n_machines):
            c_val = c[j]
            if prev_c > c_val:
                c_val = prev_c
            c_val += p_row[j]
            c[j] = c_val
            prev_c = c_val
    return c[-1]

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # --- Phase 1: Priority-based Constructive Heuristic ---
    # Insert jobs one at a time, choosing the position that minimizes makespan
    # using the exact critical path recurrence
    
    # Sort jobs by a priority key: total processing time descending (as in NEH)
    # but we'll evaluate each insertion position exactly
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    current_order = []
    c_vec = np.zeros(n_machines)
    
    for job in sorted_jobs:
        best_pos = len(current_order)
        best_ms = float('inf')
        
        for pos in range(len(current_order) + 1):
            # Compute makespan if we insert job at position pos
            # Use the prefix completion vector at pos, then process job, then rest
            c_prefix = _completion_vector(processing_times, current_order[:pos])
            
            # Process the new job
            p_row = processing_times[job]
            c_after_job = c_prefix.copy()
            prev_c = 0.0
            for j in range(n_machines):
                c_val = c_after_job[j]
                if prev_c > c_val:
                    c_val = prev_c
                c_val += p_row[j]
                c_after_job[j] = c_val
                prev_c = c_val
            
            # Process the remaining jobs from pos onwards
            ms = _makespan_from_suffix(processing_times, current_order, pos, c_after_job)
            
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        # Insert job at best position
        current_order.insert(best_pos, job)
        
        if time.time() > deadline:
            break
    
    best_order = list(current_order)
    best_ms = _makespan(processing_times, best_order)
    
    # --- Phase 2: Large Neighborhood Search with Exact Delta Evaluation ---
    # For each job, try removing it and reinserting at all positions
    # Use prefix completion vectors for efficient evaluation
    
    improved = True
    # Limit LS iterations for large N to avoid timeout
    if n_jobs > 100:
        max_ls_iter = 2
    elif n_jobs > 50:
        max_ls_iter = 5
    else:
        max_ls_iter = 20
    ls_iter = 0
    
    while improved and time.time() < deadline and ls_iter < max_ls_iter:
        improved = False
        ls_iter += 1
        n = len(best_order)
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            job = best_order[i]
            # Remove job from position i
            temp_order = best_order[:i] + best_order[i+1:]
            
            # Precompute prefix completion vectors for temp_order
            c_prefixes = _compute_c_prefix(processing_times, temp_order)
            
            best_pos_local = i
            best_ms_local = float('inf')
            
            for pos in range(len(temp_order) + 1):
                # Compute makespan with job inserted at pos
                c_before = c_prefixes[pos]
                
                # Process the new job
                p_row = processing_times[job]
                c_after_job = c_before.copy()
                prev_c = 0.0
                for j in range(n_machines):
                    c_val = c_after_job[j]
                    if prev_c > c_val:
                        c_val = prev_c
                    c_val += p_row[j]
                    c_after_job[j] = c_val
                    prev_c = c_val
                
                # Process remaining jobs
                ms = _makespan_from_suffix(processing_times, temp_order, pos, c_after_job)
                
                if ms < best_ms_local:
                    best_ms_local = ms
                    best_pos_local = pos
            
            if best_ms_local < best_ms - 1e-9:
                # Apply the best move
                best_order = temp_order[:best_pos_local] + [job] + temp_order[best_pos_local:]
                best_ms = best_ms_local
                improved = True
                # Restart the scan
                break
    
    # --- Phase 3: Simulated Annealing with Swap Moves (if time remains) ---
    if time.time() < deadline:
        current = list(best_order)
        current_ms = best_ms
        T = max(best_ms * 0.01, 1.0)
        cooling = 0.999
        min_T = max(best_ms * 1e-6, 1e-3)
        
        while T > min_T and time.time() < deadline:
            n = len(current)
            if n < 2:
                break
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i == j:
                T *= cooling
                continue
            
            # Swap move in-place
            current[i], current[j] = current[j], current[i]
            
            new_ms = _makespan(processing_times, current)
            delta = new_ms - current_ms
            
            if delta < 0 or random.random() < np.exp(-delta / max(T, 1e-10)):
                current_ms = new_ms
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_order = list(current)
            else:
                # Revert
                current[i], current[j] = current[j], current[i]
            
            T *= cooling
    
    return best_order
# EVOLVE-BLOCK-END
