"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _calc_makespan(processing_times, perm):
    """Calculate makespan for a given permutation using optimized Python lists."""
    n_machines = processing_times.shape[1]
    # perm is a list or array of job indices
    completion = [0.0] * n_machines
    
    for job in perm:
        # Access the row of processing times for this job
        row = processing_times[job]
        # Iterate over machines
        for m in range(n_machines):
            if m == 0:
                completion[0] += row[m]
            else:
                # completion[m] = max(completion[m], completion[m-1]) + row[m]
                c_prev = completion[m-1]
                c_curr = completion[m]
                if c_curr < c_prev:
                    c_curr = c_prev
                completion[m] = c_curr + row[m]
    
    return completion[-1]

def _compute_prefix_completions(processing_times, perm):
    """
    Compute completion times for all prefixes of the permutation.
    Returns a list of lists, where prefix_comp[k][m] is the completion time
    on machine m after processing the first k jobs in perm.
    prefix_comp[0] is [0, 0, ..., 0].
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    prefix_comp = [[0.0] * n_machines for _ in range(n_jobs + 1)]
    
    for k in range(n_jobs):
        job = perm[k]
        row = processing_times[job]
        prev_comp = prefix_comp[k]
        curr_comp = prefix_comp[k+1]
        
        for m in range(n_machines):
            if m == 0:
                curr_comp[0] = prev_comp[0] + row[0]
            else:
                # curr_comp[m] = max(prev_comp[m], curr_comp[m-1]) + row[m]
                c_prev_m = prev_comp[m]
                c_curr_m_minus_1 = curr_comp[m-1]
                max_c = c_prev_m if c_prev_m > c_curr_m_minus_1 else c_curr_m_minus_1
                curr_comp[m] = max_c + row[m]
                
    return prefix_comp

def _eval_insertion(processing_times, perm, i, j, prefix_comp):
    """
    Evaluate makespan of sequence formed by removing job at i and inserting at j.
    Uses prefix_comp to speed up calculation.
    i: index of job to remove
    j: index where job is inserted (in the original sequence indices)
    Note: If i < j, the job moves from i to j. The jobs i+1..j-1 shift left.
          If i > j, the job moves from i to j. The jobs j..i-1 shift right.
          If i == j, return current makespan.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    if i == j:
        return prefix_comp[n_jobs][n_machines-1]
        
    job_to_move = perm[i]
    row = processing_times[job_to_move]
    
    if i < j:
        # Start state: after job i-1
        # If i=0, start state is prefix_comp[0]
        start_state = prefix_comp[i]
        
        # Process jobs i+1 to j-1
        # These jobs are shifted left by 1 in the new sequence relative to their original positions?
        # Original: ... P[i-1], P[i], P[i+1], ..., P[j-1], P[j] ...
        # New: ... P[i-1], P[i+1], ..., P[j-1], P[i], P[j] ...
        
        # We start with state after P[i-1].
        curr_state = list(start_state) # Copy
        
        # Process P[i+1] ... P[j-1]
        for k in range(i + 1, j):
            job = perm[k]
            r = processing_times[job]
            for m in range(n_machines):
                if m == 0:
                    curr_state[0] += r[0]
                else:
                    c_prev = curr_state[m-1]
                    c_curr = curr_state[m]
                    if c_curr < c_prev:
                        c_curr = c_prev
                    curr_state[m] = c_curr + r[m]
                    
        # Now insert P[i]
        for m in range(n_machines):
            if m == 0:
                curr_state[0] += row[0]
            else:
                c_prev = curr_state[m-1]
                c_curr = curr_state[m]
                if c_curr < c_prev:
                    c_curr = c_prev
                curr_state[m] = c_curr + row[m]
                
        # Now process P[j] ... P[n-1]
        # Note: In the original sequence, P[j] was at index j.
        # In the new sequence, P[j] is still at index j (since we removed i<j and inserted i at j, 
        # the length of prefix j is same, but content changed).
        # Wait.
        # Original indices: 0 ... i-1, i, i+1 ... j-1, j, j+1 ... n-1
        # New indices: 0 ... i-1, i+1 ... j-1, i, j, j+1 ... n-1
        # The jobs P[j], P[j+1]... are at the same indices in the new sequence?
        # Let's check.
        # New seq: [0..i-1] (len i), [i+1..j-1] (len j-i-1), [i] (len 1), [j..n-1] (len n-j)
        # Total len before P[j] in new seq: i + (j-i-1) + 1 = j.
        # So P[j] is at index j in the new sequence.
        # Therefore, we can use the "tail" of the original sequence starting from j.
        # But the state entering P[j] is different.
        # So we just simulate from j to n-1 using curr_state.
        
        for k in range(j, n_jobs):
            job = perm[k]
            r = processing_times[job]
            for m in range(n_machines):
                if m == 0:
                    curr_state[0] += r[0]
                else:
                    c_prev = curr_state[m-1]
                    c_curr = curr_state[m]
                    if c_curr < c_prev:
                        c_curr = c_prev
                    curr_state[m] = c_curr + r[m]
                        
        return curr_state[n_machines-1]
        
    else: # i > j
        # Original: ... P[j-1], P[j], ..., P[i-1], P[i], P[i+1] ...
        # New: ... P[j-1], P[i], P[j], ..., P[i-1], P[i+1] ...
        
        # Start state: after job j-1
        start_state = prefix_comp[j]
        curr_state = list(start_state)
        
        # Insert P[i] first
        for m in range(n_machines):
            if m == 0:
                curr_state[0] += row[0]
            else:
                c_prev = curr_state[m-1]
                c_curr = curr_state[m]
                if c_curr < c_prev:
                    c_curr = c_prev
                curr_state[m] = c_curr + row[m]
                
        # Process P[j] ... P[i-1]
        for k in range(j, i):
            job = perm[k]
            r = processing_times[job]
            for m in range(n_machines):
                if m == 0:
                    curr_state[0] += r[0]
                else:
                    c_prev = curr_state[m-1]
                    c_curr = curr_state[m]
                    if c_curr < c_prev:
                        c_curr = c_prev
                    curr_state[m] = c_curr + r[m]
                        
        # Process P[i+1] ... P[n-1]
        # In new sequence, P[i+1] is at index i+1?
        # New prefix length before P[i+1]:
        # [0..j-1] (len j), [i] (len 1), [j..i-1] (len i-j)
        # Total: j + 1 + i - j = i + 1.
        # So P[i+1] is at index i+1 in new sequence.
        # We can simulate from i+1 to n-1.
        
        for k in range(i + 1, n_jobs):
            job = perm[k]
            r = processing_times[job]
            for m in range(n_machines):
                if m == 0:
                    curr_state[0] += r[0]
                else:
                    c_prev = curr_state[m-1]
                    c_curr = curr_state[m]
                    if c_curr < c_prev:
                        c_curr = c_prev
                    curr_state[m] = c_curr + r[m]
                    
        return curr_state[n_machines-1]


def _neh_heuristic(processing_times):
    """
    NEH heuristic: 
    1. Sort jobs by total processing time descending.
    2. Insert each job into the best position in the current sequence.
    """
    n_jobs, n_machines = processing_times.shape
    
    # Sort by total processing time descending
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    # Start with the first job
    sequence = [order[0]]
    prefix_comp = _compute_prefix_completions(processing_times, sequence)
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_makespan = float('inf')
        
        # Try inserting at each position
        for pos in range(len(sequence) + 1):
            # Evaluate insertion of job at pos
            # We can use a modified _eval_insertion or just compute directly
            # Since we are building the sequence, we don't have a "remove" index.
            # We are inserting a new job.
            # This is slightly different from _eval_insertion which moves an existing job.
            # For NEH, we can just compute makespan of new_seq directly.
            # Or we can adapt _eval_insertion logic.
            # Let's just use _calc_makespan for NEH as it's O(N^2*M) total which is fine for N~100.
            # N=100 -> 100*100*10 = 100k ops. Very fast.
            new_seq = sequence[:pos] + [job] + sequence[pos:]
            ms = _calc_makespan(processing_times, new_seq)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence.insert(best_pos, job)
        # Update prefix_comp? No, we just recomputed for the new sequence implicitly.
        # We don't need prefix_comp for NEH if we use _calc_makespan.
        # But for Local Search, we will compute prefix_comp.
    
    return sequence


def _local_search(processing_times, sequence, deadline):
    """
    Insertion local search: try moving a job to a different position.
    Use a time limit to stop before deadline.
    """
    n_jobs = len(sequence)
    current_seq = list(sequence)
    current_makespan = _calc_makespan(processing_times, current_seq)
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Compute prefix completions for current sequence
        prefix_comp = _compute_prefix_completions(processing_times, current_seq)
        
        # Try all pairs of positions (i, j)
        # Move job at i to position j
        for i in range(n_jobs):
            if time.time() >= deadline:
                break
            job = current_seq[i]
            
            # We need to try inserting at all positions j != i
            # Optimization: Check j in a specific order? Or just all j.
            # To find the best move, we might want to check all.
            # But we stop at first improvement (first-improvement strategy) or best improvement?
            # First improvement is faster per iteration but might take more iterations.
            # Best improvement requires checking all j for each i, then picking the best.
            # Let's stick to first improvement for speed, or best improvement if it converges faster.
            # Given the speedup, best improvement might be viable.
            # However, implementing best improvement requires storing the best move found in the pass.
            # Let's try first improvement first to keep it simple and fast.
            
            for j in range(n_jobs):
                if j == i:
                    continue
                if time.time() >= deadline:
                    break
                
                new_makespan = _eval_insertion(processing_times, current_seq, i, j, prefix_comp)
                
                if new_makespan < current_makespan:
                    # Perform the move on current_seq
                    job_to_move = current_seq.pop(i)
                    if j > i:
                        insert_idx = j - 1
                    else:
                        insert_idx = j
                    current_seq.insert(insert_idx, job_to_move)
                    current_makespan = new_makespan
                    improved = True
                    # Restart i loop to ensure we explore from new state
                    break
            
            if improved:
                break
    
    return current_seq, current_makespan


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    # Step 1: NEH heuristic
    initial_seq = _neh_heuristic(processing_times)
    
    # Step 2: Local search with time limit
    best_seq, best_makespan = _local_search(processing_times, initial_seq, deadline)
    
    return best_seq


# EVOLVE-BLOCK-END
