"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


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
            ms = _makespan(new_seq, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        # Insert the job at the best position
        current_sequence.insert(best_pos, job)
        
    return current_sequence


def _oropt_makespan_fast(order, processing_times, block_start, block_size, new_pos):
    """
    Fast makespan evaluation for an Or-opt move:
    - Remove block of length block_size starting at block_start
    - Insert it at position new_pos in the resulting list
    Uses incremental computation to avoid full O(n*m) recalculation.
    """
    n = len(order)
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
    
    # Build the new order
    block = order[block_start:block_start + block_size]
    remaining = order[:block_start] + order[block_start + block_size:]
    
    # new_pos is the insertion index in 'remaining'
    # Clamp new_pos to valid range
    if new_pos < 0:
        new_pos = 0
    if new_pos > len(remaining):
        new_pos = len(remaining)
    
    new_order = remaining[:new_pos] + block + remaining[new_pos:]
    
    # Compute makespan for new_order
    prev_completion = np.zeros(n_machines)
    
    for j in new_order:
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
    current_makespan = _makespan(current_order, processing_times)
    
    best_order = current_order[:]
    best_makespan = current_makespan
    
    # 2. Or-opt Local Search with fast evaluation
    # We perform iterative improvement using Or-opt moves.
    # An Or-opt move takes a block of 1, 2, or 3 jobs and inserts it at a new position.
    # We do this until no improvement is found in a full pass or time runs out.
    
    improved = True
    no_improve_count = 0
    max_no_improve = 3  # Stop after 3 full passes with no improvement
    
    while improved and time.time() < deadline:
        improved = False
        n = len(current_order)
        
        for block_size in [1, 2, 3]:
            if time.time() > deadline:
                return best_order
            
            for i in range(n - block_size + 1):
                if time.time() > deadline:
                    return best_order
                
                block = current_order[i:i+block_size]
                
                best_pos = -1
                best_ms = current_makespan
                
                # Evaluate all insertion positions
                for j in range(n - block_size + 1):
                    if j == i:
                        continue
                    
                    ms = _oropt_makespan_fast(current_order, processing_times, i, block_size, j)
                    
                    if ms < best_ms:
                        best_ms = ms
                        best_pos = j
                        
                if best_pos != -1:
                    # Apply the best move found for this block_size and i
                    remaining = current_order[:i] + current_order[i+block_size:]
                    current_order = remaining[:best_pos] + block + remaining[best_pos:]
                    current_makespan = best_ms
                    
                    if current_makespan < best_makespan:
                        best_makespan = current_makespan
                        best_order = current_order[:]
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
    
    # 3. Final polish: single-job insertion moves with tighter tolerance
    if time.time() < deadline - 2.0:
        polish_order = best_order[:]
        polish_ms = best_makespan
        n = len(polish_order)
        
        for _ in range(3):  # A few polish iterations
            if time.time() > deadline:
                break
            any_improvement = False
            for i in range(n):
                if time.time() > deadline:
                    break
                job = polish_order[i]
                remaining = polish_order[:i] + polish_order[i+1:]
                best_pos = i
                best_pms = polish_ms
                
                for j in range(n):
                    if j == i:
                        continue
                    new_order = remaining[:j] + [job] + remaining[j:]
                    pms = _makespan(new_order, processing_times)
                    if pms < best_pms:
                        best_pms = pms
                        best_pos = j
                
                if best_pos != i:
                    remaining = polish_order[:i] + polish_order[i+1:]
                    polish_order = remaining[:best_pos] + [job] + remaining[best_pos:]
                    polish_ms = best_pms
                    any_improvement = True
                    break
            
            if not any_improvement:
                break
        
        if polish_ms < best_makespan:
            best_makespan = polish_ms
            best_order = polish_order[:]
    
    return best_order

# EVOLVE-BLOCK-END
