"""Alternative strategy: Greedy Iterated Local Search (ILS) with fast delta-evaluation for insertion moves."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(order, P):
    """
    Compute makespan for a given job order on processing_times matrix P.
    """
    n_machines = P.shape[1]
    if n_machines == 0:
        return 0.0
    
    c_machines = [0.0] * n_machines
    
    for j in order:
        row = P[j]
        c_curr = 0.0
        for m in range(n_machines):
            if c_machines[m] > c_curr:
                c_curr = c_machines[m]
            c_curr += row[m]
            c_machines[m] = c_curr
            
    return c_machines[-1]

def _delta_insertion(P, seq, job, pos):
    """
    Calculate the change in makespan if 'job' is inserted at 'pos' in 'seq'.
    Assumes 'job' is NOT in 'seq'.
    Returns the delta (new_makespan - old_makespan).
    """
    n_jobs = len(seq)
    n_machines = P.shape[1]
    if n_machines == 0:
        return 0.0
    
    # 1. Compute completion times for the current sequence 'seq'
    # C[j][m] is the completion time of job seq[j] on machine m
    C = np.zeros((n_jobs, n_machines))
    for j in range(n_jobs):
        job_idx = seq[j]
        for m in range(n_machines):
            if m == 0:
                C[j, m] = C[j-1, m] + P[job_idx, m] if j > 0 else P[job_idx, m]
            else:
                C[j, m] = max(C[j, m-1], C[j-1, m]) + P[job_idx, m] if j > 0 else P[job_idx, m]
                
    old_ms = C[-1, -1] if n_jobs > 0 else 0.0
    
    # 2. Calculate new completion times for the modified sequence
    # We don't need to build the full new C matrix, just the makespan.
    # The sequence is: seq[0:pos] + [job] + seq[pos:n_jobs]
    
    # Case: pos == n_jobs (append to end)
    # The last machine completion for the new job depends on the previous job on the same machine (last of seq)
    # and the previous machine completion for the new job.
    
    # We can simulate the flow for the new job and the tail.
    # Actually, a full simulation of the new sequence is O(N*M).
    # To be faster, we can try to reuse C.
    
    # Let's stick to a slightly optimized full simulation for the new sequence,
    # but we can optimize the prefix if pos is small? No, the insertion affects all subsequent jobs on all machines.
    
    # However, we can compute the new makespan efficiently by iterating through the new sequence.
    # Since we already have C for the old sequence, can we update it?
    # Inserting at pos shifts seq[pos:] to pos+1:.
    # The completion times for jobs before pos are unchanged.
    # The new job at pos depends on job at pos-1 (old) and its own previous machine.
    # Jobs after pos depend on the new job and their old predecessor (which is now shifted).
    
    # Let's just do the O(N*M) calculation for the new sequence.
    # It is still much faster than the O(N*M) full makespan if we avoid Python loops?
    # The previous _makespan uses Python loops.
    # Let's use NumPy vectorization for the makespan calculation to speed up the local search.
    
    def _fast_makespan(order, P):
        if len(order) == 0:
            return 0.0
        n_m = P.shape[1]
        # C is (n_jobs, n_machines)
        # C[j, m] = max(C[j, m-1], C[j-1, m]) + P[order[j], m]
        # We can compute this row by row.
        C = np.zeros((len(order), n_m))
        for j in range(len(order)):
            job_idx = order[j]
            if j == 0:
                C[j, 0] = P[job_idx, 0]
                if n_m > 1:
                    C[j, 1:] = P[job_idx, 1:] + P[job_idx, 0] # Wait, C[j, m] = C[j, m-1] + P[j, m]
                    # C[j, 1] = C[j, 0] + P[j, 1]
                    # C[j, 2] = C[j, 1] + P[j, 2] ...
                    # So C[j, :] is the cumulative sum of P[job_idx, :]
                    C[j, :] = np.cumsum(P[job_idx, :])
            else:
                prev_C = C[j-1, :]
                curr_P = P[job_idx, :]
                # C[j, m] = max(C[j, m-1], prev_C[m]) + curr_P[m]
                # This is a recurrence along m.
                # C[j, 0] = prev_C[0] + curr_P[0] (since C[j, -1] is 0)
                # C[j, 1] = max(C[j, 0], prev_C[1]) + curr_P[1]
                # This cannot be easily vectorized in one go without a loop or specific tricks.
                # Given N is likely small (< 100), a Python loop over m is fine, or we stick to the previous _makespan.
                # Let's just use the previous _makespan but optimized with local variables.
                pass
        return C[-1, -1]

    # The previous _makespan is simple and likely fast enough for N<100.
    # Let's just call _makespan on the new list.
    new_seq = seq[:pos] + [job] + seq[pos:]
    new_ms = _makespan(new_seq, P)
    return new_ms - old_ms

def _neh_heuristic(P):
    """
    NEH (Nawaz-Enscore-Ham) heuristic.
    """
    n_jobs, n_machines = P.shape
    totals = P.sum(axis=1)
    order_indices = np.argsort(-totals)
    
    current_seq = []
    for idx in order_indices:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [idx] + current_seq[pos:]
            ms = _makespan(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_seq.insert(best_pos, idx)
        
    return current_seq

def _local_search(order, P):
    """
    Greedy local search using insertion moves.
    Returns improved order and makespan.
    """
    n_jobs = len(order)
    current_order = order[:]
    current_ms = _makespan(current_order, P)
    improved = True
    
    while improved:
        improved = False
        best_delta = 0
        best_move = None
        
        for i in range(n_jobs):
            job = current_order[i]
            # Remove job from current_order
            # We need to evaluate insertion at all positions in the remaining list
            # To avoid modifying current_order during evaluation, we work with indices
            
            # The remaining sequence is current_order[:i] + current_order[i+1:]
            # We evaluate inserting 'job' at pos in 0..n_jobs-1 (size of remaining is n_jobs-1, so pos 0..n_jobs-1)
            
            # Optimization: Only check positions that are likely to improve?
            # For now, check all.
            
            for pos in range(n_jobs):
                # pos is the index in the NEW sequence where 'job' will be.
                # The new sequence length is n_jobs.
                # If pos == i, it's the same position (no move).
                if pos == i:
                    continue
                    
                # Construct new order for evaluation?
                # To save time, we can calculate delta without constructing the list if we had a fast delta function.
                # But _delta_insertion currently calls _makespan which constructs the list.
                # Let's just construct the list. It's O(N).
                
                # new_order = current_order[:i] + current_order[i+1:pos] + [job] + current_order[pos:i+1] ?
                # Wait, if pos > i, the indices shift.
                # Let's be careful.
                # Original: [0, 1, 2, 3, 4]
                # Remove i=2 (job 2): [0, 1, 3, 4]
                # Insert at pos=4 (end): [0, 1, 3, 4, 2]
                # Insert at pos=0 (start): [2, 0, 1, 3, 4]
                
                # Let's just create the new list.
                new_order = current_order[:]
                del new_order[i]
                new_order.insert(pos, job)
                
                new_ms = _makespan(new_order, P)
                delta = new_ms - current_ms
                
                if delta < best_delta:
                    best_delta = delta
                    best_move = (i, pos, new_order)
                    
        if best_move is not None:
            current_order = best_move[2]
            current_ms += best_delta
            improved = True
            
    return current_order, current_ms

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using NEH + Iterated Local Search.
    """
    P = np.asarray(processing_times, dtype=float)
    if P.size == 0:
        return []
        
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    start_time = time.time()
    time_limit = 145.0  # Safety margin
    
    # Initialize with NEH
    best_order = _neh_heuristic(P)
    best_ms = _makespan(best_order, P)
    
    rng = random.Random(42)
    
    # Iterated Local Search
    # Each iteration:
    # 1. Local Search from current solution
    # 2. If improved, update best
    # 3. Perturb the solution (e.g., random re-insertion of a few jobs)
    # 4. Repeat
    
    current_order = best_order[:]
    
    while (time.time() - start_time) < time_limit:
        # Local Search
        ls_order, ls_ms = _local_search(current_order, P)
        
        if ls_ms < best_ms:
            best_ms = ls_ms
            best_order = ls_order[:]
            current_order = ls_order[:]
        else:
            # No improvement, perturb the best solution to escape local optimum
            # Perturbation: pick k random jobs and re-insert them at random positions
            # k is typically small, e.g., 2-5
            k = min(n_jobs, rng.randint(2, 5))
            
            current_order = best_order[:]
            
            # Remove k random jobs
            indices_to_remove = rng.sample(range(n_jobs), k)
            # Sort indices to remove in reverse order to maintain indices
            indices_to_remove.sort(reverse=True)
            
            removed_jobs = []
            for idx in indices_to_remove:
                removed_jobs.append(current_order.pop(idx))
            
            # Re-insert them at random positions
            for job in removed_jobs:
                pos = rng.randint(0, len(current_order))
                current_order.insert(pos, job)
                
    return best_order
# EVOLVE-BLOCK-END
