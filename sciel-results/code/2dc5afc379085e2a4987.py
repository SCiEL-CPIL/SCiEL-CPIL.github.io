"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(order, processing_times):
    """
    Calculate the makespan for a given job order in a flow shop.
    order: list of job indices
    processing_times: (n_jobs, n_machines) array
    Returns the makespan (total completion time on the last machine).
    """
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
    
    # C[m] is the completion time of the current job on machine m
    prev_completion = np.zeros(n_machines)
    
    for j in order:
        curr_completion_0 = prev_completion[0] + processing_times[j, 0]
        
        if n_machines == 1:
            prev_completion[0] = curr_completion_0
            continue
            
        curr_completion = np.empty(n_machines)
        curr_completion[0] = curr_completion_0
        
        for m in range(1, n_machines):
            curr_completion[m] = max(curr_completion[m-1], prev_completion[m]) + processing_times[j, m]
            
        prev_completion = curr_completion
        
    return prev_completion[-1]


def _makespan_fast(order, pt):
    """
    Fast makespan using list-based processing times for speed.
    """
    n_machines = len(pt[0]) if len(pt) > 0 else 0
    if n_machines == 0:
        return 0.0
    
    prev = [0.0] * n_machines
    
    for j in order:
        row = pt[j]
        c0 = prev[0] + row[0]
        
        if n_machines == 1:
            prev[0] = c0
            continue
        
        curr = [0.0] * n_machines
        curr[0] = c0
        
        for m in range(1, n_machines):
            curr[m] = max(curr[m-1], prev[m]) + row[m]
        
        prev = curr
    
    return prev[-1]


def _makespan_array(order, processing_times):
    """
    Vectorized makespan calculation returning completion times per machine.
    """
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return np.array([0.0])
    
    prev_completion = np.zeros(n_machines)
    
    for j in order:
        curr_completion_0 = prev_completion[0] + processing_times[j, 0]
        
        if n_machines == 1:
            prev_completion[0] = curr_completion_0
            continue
            
        curr_completion = np.empty(n_machines)
        curr_completion[0] = curr_completion_0
        
        for m in range(1, n_machines):
            curr_completion[m] = max(curr_completion[m-1], prev_completion[m]) + processing_times[j, m]
            
        prev_completion = curr_completion
        
    return prev_completion


def _neh_heuristic(processing_times):
    """
    Generate an initial permutation using the NEH (Nawaz-Enscore-Ham) heuristic.
    """
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Convert to list of lists for faster access
    pt_list = processing_times.tolist()
    
    # Calculate sum of processing times for each job
    job_sums = processing_times.sum(axis=1)
    
    # Sort jobs in descending order of their sums
    sorted_jobs = np.argsort(-job_sums)
    
    current_sequence = []
    
    for job in sorted_jobs:
        # Find the best position to insert the current job
        best_pos = 0
        best_ms = float('inf')
        
        # Try inserting at every possible position (0 to len(current_sequence))
        for pos in range(len(current_sequence) + 1):
            new_seq = current_sequence[:pos] + [job] + current_sequence[pos:]
            ms = _makespan_fast(new_seq, pt_list)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        # Insert the job at the best position
        current_sequence.insert(best_pos, job)
        
    return current_sequence


def _compute_prefix_completions(order, processing_times):
    """
    Precompute prefix completion times for efficient incremental evaluation.
    Returns a list where prefix[i] is the completion vector after processing jobs order[0:i].
    """
    n = len(order)
    n_machines = processing_times.shape[1]
    
    if n_machines == 0:
        return [np.array([0.0])] * (n + 1)
    
    prefix = [np.zeros(n_machines)]
    current = np.zeros(n_machines)
    
    for i in range(n):
        j = order[i]
        curr_completion_0 = current[0] + processing_times[j, 0]
        
        if n_machines == 1:
            current[0] = curr_completion_0
        else:
            curr_completion = np.empty(n_machines)
            curr_completion[0] = curr_completion_0
            
            for m in range(1, n_machines):
                curr_completion[m] = max(curr_completion[m-1], current[m]) + processing_times[j, m]
            
            current = curr_completion
        
        prefix.append(current.copy())
    
    return prefix


def _eval_insertion(order, processing_times, prefix, i, j):
    """
    Evaluate makespan when moving job at position i to position j (i < j case).
    Uses precomputed prefix completions for O(n*m) instead of O(n*m) full recompute.
    """
    n = len(order)
    n_machines = processing_times.shape[1]
    
    if n_machines == 0:
        return 0.0
    
    # Remove job at i, insert at j
    # New order: order[0:i] + order[i+1:j+1] + [order[i]] + order[j+1:]
    
    # We need to recompute from position i onwards
    # Start with completion state after processing order[0:i]
    current = prefix[i].copy()
    
    # Process jobs from i+1 to j (these shift left by one)
    for k in range(i + 1, j + 1):
        job = order[k]
        curr_completion_0 = current[0] + processing_times[job, 0]
        
        if n_machines == 1:
            current[0] = curr_completion_0
        else:
            curr_completion = np.empty(n_machines)
            curr_completion[0] = curr_completion_0
            
            for m in range(1, n_machines):
                curr_completion[m] = max(curr_completion[m-1], current[m]) + processing_times[job, m]
            
            current = curr_completion
    
    # Process the moved job
    job = order[i]
    curr_completion_0 = current[0] + processing_times[job, 0]
    
    if n_machines == 1:
        current[0] = curr_completion_0
    else:
        curr_completion = np.empty(n_machines)
        curr_completion[0] = curr_completion_0
        
        for m in range(1, n_machines):
            curr_completion[m] = max(curr_completion[m-1], current[m]) + processing_times[job, m]
        
        current = curr_completion
    
    # Process remaining jobs from j+1 to end
    for k in range(j + 1, n):
        job = order[k]
        curr_completion_0 = current[0] + processing_times[job, 0]
        
        if n_machines == 1:
            current[0] = curr_completion_0
        else:
            curr_completion = np.empty(n_machines)
            curr_completion[0] = curr_completion_0
            
            for m in range(1, n_machines):
                curr_completion[m] = max(curr_completion[m-1], current[m]) + processing_times[job, m]
            
            current = curr_completion
    
    return current[-1]


def _eval_insertion_reverse(order, processing_times, prefix, i, j):
    """
    Evaluate makespan when moving job at position i to position j (i > j case).
    """
    n = len(order)
    n_machines = processing_times.shape[1]
    
    if n_machines == 0:
        return 0.0
    
    # Remove job at i, insert at j
    # New order: order[0:j] + [order[i]] + order[j:i] + order[i+1:]
    
    # Start with completion state after processing order[0:j]
    current = prefix[j].copy()
    
    # Process the moved job
    job = order[i]
    curr_completion_0 = current[0] + processing_times[job, 0]
    
    if n_machines == 1:
        current[0] = curr_completion_0
    else:
        curr_completion = np.empty(n_machines)
        curr_completion[0] = curr_completion_0
        
        for m in range(1, n_machines):
            curr_completion[m] = max(curr_completion[m-1], current[m]) + processing_times[job, m]
        
        current = curr_completion
    
    # Process jobs from j to i-1 (these shift right by one)
    for k in range(j, i):
        job = order[k]
        curr_completion_0 = current[0] + processing_times[job, 0]
        
        if n_machines == 1:
            current[0] = curr_completion_0
        else:
            curr_completion = np.empty(n_machines)
            curr_completion[0] = curr_completion_0
            
            for m in range(1, n_machines):
                curr_completion[m] = max(curr_completion[m-1], current[m]) + processing_times[job, m]
            
            current = curr_completion
    
    # Process remaining jobs from i+1 to end
    for k in range(i + 1, n):
        job = order[k]
        curr_completion_0 = current[0] + processing_times[job, 0]
        
        if n_machines == 1:
            current[0] = curr_completion_0
        else:
            curr_completion = np.empty(n_machines)
            curr_completion[0] = curr_completion_0
            
            for m in range(1, n_machines):
                curr_completion[m] = max(curr_completion[m-1], current[m]) + processing_times[job, m]
            
            current = curr_completion
    
    return current[-1]


def _local_search_insertion(order, processing_times, deadline, best_order, best_makespan):
    """
    Local search using insertion moves with prefix precomputation.
    Returns (best_order, best_makespan)
    """
    n = len(order)
    n_machines = processing_times.shape[1]
    
    if n <= 1 or n_machines == 0:
        return order[:], best_makespan
    
    pt_list = processing_times.tolist()
    current_order = order[:]
    current_ms = best_makespan
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Precompute prefix completions
        prefix = _compute_prefix_completions(current_order, processing_times)
        
        for i in range(n):
            if time.time() > deadline:
                break
            
            job = current_order[i]
            
            # Try all insertion positions
            best_pos = i
            best_eval_ms = current_ms
            
            for j in range(n):
                if j == i:
                    continue
                
                if i < j:
                    ms = _eval_insertion(current_order, processing_times, prefix, i, j)
                else:
                    ms = _eval_insertion_reverse(current_order, processing_times, prefix, i, j)
                
                if ms < best_eval_ms:
                    best_eval_ms = ms
                    best_pos = j
            
            if best_pos != i:
                # Apply the move
                job = current_order.pop(i)
                current_order.insert(best_pos, job)
                current_ms = best_eval_ms
                
                if current_ms < best_makespan:
                    best_makespan = current_ms
                    best_order = current_order[:]
                
                improved = True
                break  # Restart the outer loop with new prefix
    
    return best_order, best_makespan


def _polish_pass(order, processing_times, deadline):
    """
    High-precision polish pass with aggressive neighborhood exploration.
    Uses multiple restarts and tighter convergence criteria.
    """
    n = len(order)
    n_machines = processing_times.shape[1]
    
    if n <= 1 or n_machines == 0:
        return order[:], _makespan(order, processing_times)
    
    pt_list = processing_times.tolist()
    best_order = order[:]
    best_ms = _makespan_fast(best_order, pt_list)
    
    # Multiple short restarts to escape local minima
    n_restarts = 3
    time_per_restart = max(1.0, (deadline - time.time()) / n_restarts)
    
    for restart in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Start from current best with small random perturbation
        current_order = best_order[:]
        
        if restart > 0:
            # Perturb: swap two random positions
            if n >= 2:
                i = random.randint(0, n - 2)
                j = random.randint(i + 1, n - 1)
                current_order[i], current_order[j] = current_order[j], current_order[i]
        
        current_ms = _makespan_fast(current_order, pt_list)
        
        # Intensive local search within time budget
        improved = True
        while improved and time.time() < deadline - 0.5:
            improved = False
            
            prefix = _compute_prefix_completions(current_order, processing_times)
            
            for i in range(n):
                if time.time() > deadline - 0.5:
                    break
                
                best_pos = i
                best_eval_ms = current_ms
                
                for j in range(n):
                    if j == i:
                        continue
                    
                    if i < j:
                        ms = _eval_insertion(current_order, processing_times, prefix, i, j)
                    else:
                        ms = _eval_insertion_reverse(current_order, processing_times, prefix, i, j)
                    
                    if ms < best_eval_ms:
                        best_eval_ms = ms
                        best_pos = j
                
                if best_pos != i:
                    job = current_order.pop(i)
                    current_order.insert(best_pos, job)
                    current_ms = best_eval_ms
                    
                    if current_ms < best_ms:
                        best_ms = current_ms
                        best_order = current_order[:]
                    
                    improved = True
                    break
    
    return best_order, best_ms


def _simulated_annealing(order, processing_times, deadline, best_order, best_makespan):
    """
    Simulated annealing with insertion moves to escape local minima.
    """
    n = len(order)
    n_machines = processing_times.shape[1]
    
    if n <= 1 or n_machines == 0:
        return best_order, best_makespan
    
    pt_list = processing_times.tolist()
    current_order = order[:]
    current_ms = _makespan_fast(current_order, pt_list)
    
    # Initial temperature based on problem size
    T = max(1.0, current_ms * 0.1)
    T_min = max(0.01, current_ms * 0.001)
    cooling_rate = 0.97  # Slower cooling for more exploration
    
    while T > T_min and time.time() < deadline:
        # Generate a random insertion move
        i = random.randint(0, n - 1)
        j = random.randint(0, n - 1)
        
        if i == j:
            continue
        
        # Compute makespan of the new solution
        job = current_order[i]
        remaining = current_order[:i] + current_order[i+1:]
        
        # Adjust j if necessary
        if j > i:
            j -= 1
        j = max(0, min(j, len(remaining)))
        
        new_order = remaining[:j] + [job] + remaining[j:]
        new_ms = _makespan_fast(new_order, pt_list)
        
        delta = new_ms - current_ms
        
        # Accept or reject
        if delta < 0:
            # Always accept improvements
            current_order = new_order
            current_ms = new_ms
            
            if current_ms < best_makespan:
                best_makespan = current_ms
                best_order = current_order[:]
        else:
            # Accept with probability exp(-delta/T)
            if random.random() < np.exp(-delta / T):
                current_order = new_order
                current_ms = new_ms
        
        # Cool down
        T *= cooling_rate
    
    return best_order, best_makespan


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    # 1. Generate initial solution using NEH heuristic
    current_order = _neh_heuristic(processing_times)
    current_makespan = _makespan_fast(current_order, processing_times.tolist())
    
    best_order = current_order[:]
    best_makespan = current_makespan
    
    # 2. Local search with insertion moves
    best_order, best_makespan = _local_search_insertion(
        best_order, processing_times, deadline, best_order, best_makespan
    )
    
    # 3. Simulated annealing to escape local minima
    if time.time() < deadline - 10.0:
        best_order, best_makespan = _simulated_annealing(
            best_order, processing_times, deadline, best_order, best_makespan
        )
    
    # 4. High-precision polish pass with multiple restarts
    if time.time() < deadline - 2.0:
        best_order, best_makespan = _polish_pass(
            best_order, processing_times, deadline
        )
    
    return best_order

# EVOLVE-BLOCK-END
