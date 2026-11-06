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
    if not order:
        return 0.0
    return _makespan_suffix(order, processing_times, 0, np.zeros(processing_times.shape[1]))

def _makespan_suffix(order, processing_times, start_idx, prev_completion):
    """
    Calculate the makespan for the suffix of order starting from start_idx,
    given the completion times of the job at start_idx-1.
    """
    n_machines = processing_times.shape[1]
    if n_machines == 0:
        return 0.0
    
    # prev_completion is a copy of the state before start_idx
    # We need to update it as we process jobs from start_idx to end
    curr_completion = prev_completion.copy()
    
    for idx in range(start_idx, len(order)):
        j = order[idx]
        
        curr_completion[0] = curr_completion[0] + processing_times[j, 0]
        
        if n_machines == 1:
            continue
            
        for m in range(1, n_machines):
            curr_completion[m] = max(curr_completion[m-1], curr_completion[m]) + processing_times[j, m]
            
    return curr_completion[-1]


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
    
    # 2. Or-opt Local Search with Fast Suffix Evaluation
    improved = True
    while improved and time.time() < deadline:
        improved = False
        n = len(current_order)
        n_machines = processing_times.shape[1]
        
        # Precompute prefix completion times for current_order
        # prefix_comp[k] is the completion time array after processing current_order[k]
        # We need the state BEFORE index k, which is prefix_comp[k-1] or zeros.
        # Let's store the state AFTER each job.
        # states[0] = completion after job 0
        # states[k] = completion after job k
        # To evaluate suffix starting at idx, we need state BEFORE idx.
        # If idx == 0, state is zeros.
        # If idx > 0, state is states[idx-1].
        
        states = [None] * n
        curr_state = np.zeros(n_machines)
        for idx in range(n):
            j = current_order[idx]
            curr_state[0] = curr_state[0] + processing_times[j, 0]
            if n_machines > 1:
                for m in range(1, n_machines):
                    curr_state[m] = max(curr_state[m-1], curr_state[m]) + processing_times[j, m]
            states[idx] = curr_state.copy()
            
        for block_size in [1, 2, 3]:
            for i in range(n - block_size + 1):
                if time.time() > deadline:
                    return best_order
                
                block = current_order[i:i+block_size]
                remaining = current_order[:i] + current_order[i+block_size:]
                
                best_pos = -1
                best_ms = current_makespan
                
                for j in range(n - block_size + 1):
                    if j == i:
                        continue
                    
                    # Determine start_idx for suffix evaluation
                    # If j < i: prefix is current_order[:j], start_idx = j
                    # If j > i: prefix is current_order[:i], start_idx = i
                    # Note: j is index in 'remaining'. 
                    # If j < i, new_order starts with current_order[:j].
                    # If j >= i, new_order starts with current_order[:i].
                    
                    if j < i:
                        start_idx = j
                        # prev_state is state before j in current_order
                        if start_idx == 0:
                            prev_state = np.zeros(n_machines)
                        else:
                            prev_state = states[start_idx - 1]
                    else:
                        start_idx = i
                        if start_idx == 0:
                            prev_state = np.zeros(n_machines)
                        else:
                            prev_state = states[start_idx - 1]
                            
                    # Construct new_order for the suffix part?
                    # We need the sequence of jobs from start_idx onwards in new_order.
                    # If j < i:
                    # new_order = current_order[:j] + block + current_order[j:i] + current_order[i+block_size:]
                    # Suffix starts at j.
                    # Jobs: block, then current_order[j:i], then current_order[i+block_size:]
                    # Sequence: block + remaining[j:]  (Wait, remaining is current_order without block)
                    # remaining[j:] is current_order[j:i] + current_order[i+block_size:]
                    # So suffix_jobs = block + remaining[j:]
                    
                    # If j >= i:
                    # new_order = current_order[:i] + remaining[j-i:] + block + remaining[...]?
                    # No.
                    # remaining = current_order[:i] + current_order[i+block_size:]
                    # Insert block at j in remaining.
                    # If j >= i:
                    # The first i jobs of remaining are current_order[:i].
                    # So new_order[:i] = current_order[:i].
                    # new_order[i:] = remaining[i:j] + block + remaining[j:]
                    # Wait, j is index in remaining.
                    # remaining[0:i] is current_order[:i].
                    # remaining[i:] is current_order[i+block_size:].
                    # If we insert at j (where j >= i):
                    # new_order = remaining[:j] + block + remaining[j:]
                    # Since j >= i, remaining[:j] contains all of current_order[:i] and some of the tail.
                    # So new_order[:i] is indeed current_order[:i].
                    # The suffix starts at i.
                    # Suffix jobs: remaining[i:j] + block + remaining[j:]
                    # Note: remaining[i:j] is current_order[i+block_size : i+block_size + (j-i)]
                    
                    if j < i:
                        suffix_jobs = block + remaining[j:]
                    else:
                        suffix_jobs = remaining[i:j] + block + remaining[j:]
                        # Wait, if j == i, suffix_jobs = [] + block + remaining[i:] = block + remaining[i:]
                        # But we skip j == i.
                        
                    # Evaluate makespan of suffix_jobs with prev_state
                    ms = _makespan_suffix(suffix_jobs, processing_times, 0, prev_state)
                    
                    if ms < best_ms:
                        best_ms = ms
                        best_pos = j
                        
                if best_pos != -1:
                    # Apply the best move found for this block_size and i
                    current_order = remaining[:best_pos] + block + remaining[best_pos:]
                    current_makespan = best_ms
                    
                    if current_makespan < best_makespan:
                        best_makespan = current_makespan
                        best_order = current_order[:]
                        
                    improved = True
                    break
            
            if improved:
                break
                
    return best_order

# EVOLVE-BLOCK-END
