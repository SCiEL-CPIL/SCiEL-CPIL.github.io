import numpy as np
import time

def _makespan(processing_times, order):
    """Compute makespan for a given job order on a flow shop."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0.0
    
    machine_completion = np.zeros(n_machines)
    makespan = 0.0
    
    for job_idx in order:
        prev_completion = 0.0
        for m in range(n_machines):
            start_time = max(prev_completion, machine_completion[m])
            prev_completion = start_time + processing_times[job_idx, m]
            machine_completion[m] = prev_completion
        makespan = max(makespan, prev_completion)
    
    return makespan

def _compute_completion_matrix(processing_times, order):
    """
    Compute the completion time matrix C of shape (n_jobs, n_machines).
    C[i, m] is the completion time of the job at position i on machine m.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return np.zeros((0, n_machines))
    
    C = np.zeros((n_jobs, n_machines))
    for i in range(n_jobs):
        job = order[i]
        if i == 0:
            C[0, 0] = processing_times[job, 0]
            for m in range(1, n_machines):
                C[0, m] = C[0, m-1] + processing_times[job, m]
        else:
            C[i, 0] = C[i-1, 0] + processing_times[job, 0]
            for m in range(1, n_machines):
                C[i, m] = max(C[i-1, m], C[i, m-1]) + processing_times[job, m]
    return C

def _neh_heuristic(processing_times):
    """NEH heuristic for flow shop scheduling."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    sequence = [sorted_jobs[0]]
    
    for job in sorted_jobs[1:]:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence = sequence[:best_pos] + [job] + sequence[best_pos:]
    
    return sequence

def _local_search_best_improvement(processing_times, initial_order, time_limit):
    """
    Local search using Best-Improvement strategy with incremental evaluation.
    Moves: Insertion (move job i to position j).
    """
    n_jobs = len(initial_order)
    n_machines = processing_times.shape[1]
    if n_jobs <= 1:
        return list(initial_order)
    
    current_order = list(initial_order)
    C = _compute_completion_matrix(processing_times, current_order)
    current_makespan = C[-1, -1]
    best_order = list(current_order)
    best_makespan = current_makespan
    
    start_time = time.time()
    
    while True:
        if time.time() - start_time > time_limit:
            break
            
        best_move_i = -1
        best_move_j = -1
        best_move_ms = current_makespan
        best_move_C = None
        
        # Iterate over all possible insertion moves
        # Move job at index i to index j
        # j is the index in the NEW list where the job is inserted.
        # If we remove i, the list shrinks. 
        # It's easier to think: 
        # Original: [0, 1, ..., i-1, i, i+1, ..., j-1, j, ...]
        # Remove i: [0, ..., i-1, i+1, ..., j-1, j, ...]
        # Insert at j: 
        # If j > i: The job moves from i to j. In the reduced list, the position is j-1? 
        # Let's stick to standard: 
        # Remove job at i.
        # Insert it at position k in the remaining list.
        # k can be 0 to n-1.
        # If k == i or k == i-1, it's the same position (or trivially close).
        
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            
            job = current_order[i]
            
            # Try inserting at every position k in the list *after* removing i
            # The length of the reduced list is n_jobs - 1.
            # k ranges from 0 to n_jobs - 1.
            # If k == i, the job stays at i (if we consider indices in the original list? No, indices shift).
            # Let's just generate the new order and check.
            
            for k in range(n_jobs):
                # If k == i, skip (no change)
                # If k == i-1, the job moves to i-1? 
                # Let's construct the new order.
                # Remove i:
                # new_order = current_order[:i] + current_order[i+1:]
                # Insert at k:
                # If k > i, the index in the reduced list is k-1? 
                # Actually, it's safer to just build the list.
                
                if k == i:
                    continue
                
                # Build new order
                # Remove i
                reduced = current_order[:i] + current_order[i+1:]
                
                # Determine insertion index in reduced list
                # If k <= i, insert at k.
                # If k > i, insert at k-1.
                insert_idx = k if k <= i else k - 1
                
                if insert_idx < 0:
                    continue
                if insert_idx > len(reduced):
                    insert_idx = len(reduced)
                
                new_order = reduced[:insert_idx] + [job] + reduced[insert_idx:]
                
                # Check if it's actually a change
                # If new_order == current_order, skip
                # This happens if k=i or k=i+1 (moving to adjacent position which is effectively same or swap?)
                # Actually, moving i to i+1 is a swap with i+1? No, it's a shift.
                # Let's just evaluate.
                
                # Incremental evaluation
                # Determine start index for recomputation
                start_recompute = min(i, insert_idx)
                
                # Copy prefix of C
                C_new = C.copy()
                
                # Recompute from start_recompute to end
                # We need the completion time of the job *before* start_recompute to start the recurrence.
                # If start_recompute == 0, prev_row is zeros.
                # Else, prev_row is C_new[start_recompute - 1]
                
                if start_recompute == 0:
                    prev_row = np.zeros(n_machines)
                else:
                    prev_row = C_new[start_recompute - 1].copy()
                    
                # We need to process jobs from new_order[start_recompute] to end
                # Note: The jobs in new_order[0:start_recompute] are the same as current_order[0:start_recompute]
                # So C_new[0:start_recompute] is already correct from C.
                
                for pos in range(start_recompute, n_jobs):
                    j = new_order[pos]
                    pt_row = processing_times[j]
                    
                    if pos == 0:
                        # Should not happen if start_recompute > 0
                        C_new[pos, 0] = pt_row[0]
                        for m in range(1, n_machines):
                            C_new[pos, m] = C_new[pos, m-1] + pt_row[m]
                    else:
                        # Use prev_row which is C_new[pos-1]
                        C_new[pos, 0] = prev_row[0] + pt_row[0]
                        for m in range(1, n_machines):
                            C_new[pos, m] = max(prev_row[m], C_new[pos, m-1]) + pt_row[m]
                        
                    prev_row = C_new[pos]
                    
                    # Early termination if makespan already exceeds best_move_ms
                    # The makespan is C_new[-1, -1]. 
                    # If we are at the last machine of the last job, we have the makespan.
                    # We can check if C_new[pos, n_machines-1] > best_move_ms and if this is the last job?
                    # No, later jobs might have smaller completion times? No, completion times are non-decreasing along the last machine.
                    # So if C_new[pos, n_machines-1] > best_move_ms, and pos < n_jobs-1, it might still decrease? 
                    # No, C[j, M-1] is non-decreasing with j.
                    # So if C_new[pos, n_machines-1] > best_move_ms, the final makespan will be >= this value.
                    # So we can break early.
                    if C_new[pos, n_machines - 1] > best_move_ms:
                        break
                
                new_makespan = C_new[-1, n_machines - 1]
                
                if new_makespan < best_move_ms:
                    best_move_ms = new_makespan
                    best_move_i = i
                    best_move_j = k # Store k for reconstruction? Or store new_order?
                    best_move_C = C_new
                    # We can store the new_order to avoid reconstruction later
                    # But storing a list for every better move is memory heavy.
                    # We can just store i and k, and reconstruct at the end.
                    # Or store the new_order directly since we only keep the BEST one.
                    # Let's store new_order.
                    # Actually, we only need to store the best one.
                    # So we can just update variables.
        
        if best_move_i != -1:
            # Apply the best move
            # Reconstruct new_order from best_move_i and best_move_j (k)
            # Or just use the best_move_C and re-derive order? No, C doesn't give order.
            # We need to reconstruct the order.
            i = best_move_i
            k = best_move_j
            job = current_order[i]
            reduced = current_order[:i] + current_order[i+1:]
            insert_idx = k if k <= i else k - 1
            current_order = reduced[:insert_idx] + [job] + reduced[insert_idx:]
            C = best_move_C
            current_makespan = best_move_ms
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_order = list(current_order)
        else:
            break
            
    return best_order

def _random_perturbation(order, strength=0.3, rng=None):
    """Apply random 2-opt perturbation."""
    n = len(order)
    if n <= 1:
        return list(order)
    
    if rng is None:
        rng = np.random
    
    new_order = list(order)
    num_swaps = max(1, int(strength * n))
    for _ in range(num_swaps):
        i, j = rng.choice(n, 2, replace=False)
        new_order[i], new_order[j] = new_order[j], new_order[i]
    
    return new_order

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow shop.
    """
    processing_times = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    deadline = 140.0
    start_time = time.time()
    
    # Step 1: NEH
    neh_order = _neh_heuristic(processing_times)
    best_order = list(neh_order)
    best_makespan = _makespan(processing_times, best_order)
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if remaining < 5.0:
        return best_order
    
    # Step 2: Best-Improvement Local Search
    # Allocate 60% of remaining time to this
    ls_time = min(remaining * 0.6, 60.0)
    ls_order = _local_search_best_improvement(processing_times, best_order, time_limit=ls_time)
    ls_makespan = _makespan(processing_times, ls_order)
    
    if ls_makespan < best_makespan:
        best_order = list(ls_order)
        best_makespan = ls_makespan
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if remaining < 5.0:
        return best_order
    
    # Step 3: ILS
    rng = np.random.RandomState(42)
    while time.time() - start_time < deadline - 5.0:
        # Perturb
        perturbed = _random_perturbation(best_order, strength=0.2, rng=rng)
        
        # Local search
        ls_time_r = min(5.0, deadline - time.time() - 2.0)
        if ls_time_r < 1.0:
            break
            
        ls_order_r = _local_search_best_improvement(processing_times, perturbed, time_limit=ls_time_r)
        ls_makespan_r = _makespan(processing_times, ls_order_r)
        
        if ls_makespan_r < best_makespan:
            best_order = list(ls_order_r)
            best_makespan = ls_makespan_r
            
    return best_order
