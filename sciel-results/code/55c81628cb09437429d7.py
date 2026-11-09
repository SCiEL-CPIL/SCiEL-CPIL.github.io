"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(processing_times, order):
    """
    Compute the makespan for a given job order.
    processing_times: np.ndarray of shape (n_jobs, n_machines)
    order: list of job indices
    Returns: float makespan
    """
    if len(order) == 0:
        return 0.0
    
    n_jobs = len(order)
    n_machines = processing_times.shape[1]
    
    # Extract processing times for the order
    # p[i] is the processing times for the job at position i
    p = processing_times[order]
    
    # C[j] stores the completion time of the current job on machine j
    # We iterate through jobs. For each job, we update C for all machines.
    # C_new[j] = max(C_old[j], C_new[j-1]) + p[i][j]
    
    C = np.zeros(n_machines)
    
    for i in range(n_jobs):
        # C[0] = C[0] + p[i][0]  (since C_new[-1] is effectively 0 or previous job's C[0] which is max(0, C_prev[0]) + p[0])
        # Actually, the recurrence is:
        # C[i][0] = C[i-1][0] + p[i][0]
        # C[i][j] = max(C[i-1][j], C[i][j-1]) + p[i][j]
        
        # We can update C in place from left to right.
        # Let C be the array of completion times for the previous job (i-1).
        # We want to compute C for current job i.
        # C_curr[0] = C_prev[0] + p[i][0]
        # C_curr[j] = max(C_prev[j], C_curr[j-1]) + p[i][j]
        
        prev_c = C.copy() # We need the old values for the max comparison with the column above (i-1, j)
        
        # First machine
        C[0] = prev_c[0] + p[i, 0]
        
        # Remaining machines
        for j in range(1, n_machines):
            # max(completion time of previous job on machine j, completion time of current job on previous machine)
            C[j] = max(prev_c[j], C[j-1]) + p[i, j]
            
    return C[-1]

def _neh_construction(processing_times):
    """
    Construct an initial sequence using the NEH heuristic.
    """
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    seq = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = np.inf
        
        # Try inserting job at each position in current seq
        for pos in range(len(seq) + 1):
            new_seq = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(processing_times, new_seq)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
                
        seq.insert(best_pos, job)
        
    return seq

def _insertion_local_search(processing_times, seq, time_limit, start_time, max_passes=100):
    """
    Perform Insertion-based local search (First Improvement or Best Improvement).
    We use First Improvement for speed, allowing more passes.
    """
    n_jobs = len(seq)
    current_ms = _makespan(processing_times, seq)
    
    # To speed up, we can limit the number of passes or use a time check.
    # Insertion move: pick i, j, remove seq[i], insert at j.
    # Number of moves is N*(N-1).
    
    # We will perform passes until no improvement is found or time runs out.
    # To avoid getting stuck in a cycle or taking too long on one pass, we use a fixed number of passes per "iteration" of the outer loop.
    
    improved = True
    pass_count = 0
    
    while improved and (time.time() - start_time) < time_limit - 5:
        improved = False
        pass_count += 1
        if pass_count > 50: # Safety valve, shouldn't happen often with first improvement
            break
            
        # We iterate through all possible insertion moves.
        # To make it faster, we can randomize the order or just do sequential.
        # Sequential is deterministic and fine.
        
        for i in range(n_jobs):
            if (time.time() - start_time) > time_limit - 5:
                return seq, current_ms
                
            job_to_move = seq[i]
            # Remove job
            seq.pop(i)
            
            # Try all insertion positions (0 to n_jobs-1, since we removed one, length is n_jobs-1)
            # Actually, inserting back at i is a no-op (or swap with self), so we can skip that if we want, but it's cheap.
            # We need to be careful with indices after pop.
            
            best_insert_pos = -1
            best_insert_ms = current_ms # We only accept if strictly better
            
            # We can optimize by not evaluating all positions if we find a good one quickly,
            # but for correctness in local search, we usually look for the best insertion or first improvement.
            # Let's do First Improvement for speed.
            
            # However, removing and re-inserting changes the indices.
            # It is safer to generate the new sequence and evaluate.
            
            # To save time, we can just try inserting at a few random positions or scan.
            # Scanning is O(N) evaluations, each O(N*M). Total O(N^2 * M) per pass.
            # For N=100, M=10, that's 100 * 100 * 10 = 100,000 ops per pass. Very fast.
            
            for j in range(n_jobs): # j is the insertion index in the reduced list
                # Insert at j
                seq.insert(j, job_to_move)
                new_ms = _makespan(processing_times, seq)
                
                if new_ms < best_insert_ms:
                    best_insert_ms = new_ms
                    best_insert_pos = j
                    # First Improvement: break and accept this move
                    # But we already modified seq. We need to keep this state.
                    # If we break, we accept this move.
                    break
                else:
                    # Revert insertion
                    seq.pop(j)
            
            # If we found an improvement, we have already inserted it at best_insert_pos.
            # If not, we need to insert it back at its original position i?
            # Wait, the logic above is tricky because `seq` is modified during the loop.
            
            # Let's rewrite the inner loop to be safer.
            # 1. Pop job at i.
            # 2. Try inserting at j.
            # 3. If better, keep it. Update i to be j (or adjust). Break inner loop.
            # 4. If not better, pop it back and try next j.
            # 5. If no j is better, insert back at i (original spot).
            
            # The code above does this partially but `best_insert_pos` handling is implicit.
            # If `new_ms < best_insert_ms`, we break. The job is currently at `j`.
            # We should update `current_ms` and `i`?
            # If we move the job, the indices of subsequent jobs change.
            # It's safer to restart the outer loop `i` if we made a move, or just continue carefully.
            # For simplicity and robustness, if we make a move, we can just `improved = True` and `break` the outer loop?
            # No, that's inefficient.
            
            # Let's use a simpler, safer approach for the inner loop:
            # We will not modify `seq` until we are sure.
            
            # Actually, the previous logic:
            # seq.pop(i)
            # for j...:
            #   seq.insert(j, job)
            #   eval
            #   if better: break (job stays at j)
            #   else: seq.pop(j)
            
            # If we break, the job is at j.
            # If we finish the loop without breaking, the job is NOT in seq. We must insert it back at i.
            
            if best_insert_pos == -1:
                # No improvement found, put it back at i
                seq.insert(i, job_to_move)
            else:
                # Improvement found, job is already at best_insert_pos
                current_ms = best_insert_ms
                improved = True
                # We can continue the outer loop, but the indices have shifted.
                # It's safer to restart the outer loop or just continue.
                # Continuing is fine, but `i` now points to the next element.
                # The element that was at i+1 is now at i (if j < i) or i+1 (if j > i).
                # To avoid missing moves, we can just `i -= 1` if j < i? 
                # Or simpler: just let it be. The next iteration of the outer loop will handle the new configuration.
                # However, if we moved a job, we might have skipped some potential moves in this pass.
                # That's acceptable for a heuristic.
                
    return seq, current_ms

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    """
    start_time = time.time()
    time_limit = 115  # Safety margin for 120s limit
    
    n_jobs = processing_times.shape[0]
    if n_jobs == 0:
        return []
    
    # 1. NEH Construction
    seq = _neh_construction(processing_times)
    current_ms = _makespan(processing_times, seq)
    
    # 2. Iterated Local Search
    # We will repeatedly:
    # a. Local Search (Insertion)
    # b. Perturb
    # c. Local Search again
    # d. Keep best
    
    best_seq = seq.copy()
    best_ms = current_ms
    
    # Number of ILS iterations is not fixed, but time-bounded.
    
    while (time.time() - start_time) < time_limit - 10:
        # Local Search
        seq, current_ms = _insertion_local_search(processing_times, seq, time_limit, start_time)
        
        if current_ms < best_ms:
            best_ms = current_ms
            best_seq = seq.copy()
        
        # Perturb the current local optimum
        # Number of random moves: 5-10% of N
        n_perturb = max(1, int(0.05 * n_jobs))
        for _ in range(n_perturb):
            i = random.randint(0, n_jobs - 1)
            j = random.randint(0, n_jobs - 1)
            if i != j:
                seq[i], seq[j] = seq[j], seq[i]
        
        # Check time before next local search
        if (time.time() - start_time) > time_limit - 15:
            break
            
    return best_seq
# EVOLVE-BLOCK-END
