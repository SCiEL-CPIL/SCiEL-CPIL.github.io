"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _calc_makespan(processing_times, perm):
    """
    Calculate makespan for a given permutation using a row-wise (job-by-job) loop.
    This is the standard O(N*M) algorithm with good cache locality for the completion vector.
    
    Args:
        processing_times: np.ndarray of shape (n_jobs, n_machines)
        perm: list or array of job indices in order
    
    Returns:
        makespan (float)
    """
    if len(perm) == 0:
        return 0.0
    
    n_machines = processing_times.shape[1]
    if n_machines == 1:
        return np.sum(processing_times[perm])
    
    # completion[m] is the completion time of the last processed job on machine m
    completion = np.zeros(n_machines)
    
    for job in perm:
        row = processing_times[job]
        prev_c = 0.0
        for m in range(n_machines):
            # Job can start on machine m after:
            # 1. The previous job on machine m is done (completion[m])
            # 2. The same job is done on the previous machine (prev_c)
            start_time = max(completion[m], prev_c)
            prev_c = start_time + row[m]
            completion[m] = prev_c
            
    return completion[-1]

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # 1. Initial Construction: NEH Heuristic
    # Sort jobs in decreasing order of total processing time
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    current_perm = [sorted_jobs[0]]
    
    for i in range(1, n_jobs):
        job = sorted_jobs[i]
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_perm) + 1):
            # Construct candidate permutation
            # Note: For NEH, we insert the job at every possible position and pick the best
            # This is O(N^2 * M) total for construction, which is fine for moderate N
            candidate_perm = current_perm[:pos] + [job] + current_perm[pos:]
            ms = _calc_makespan(processing_times, candidate_perm)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
                
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
        
    best_perm = current_perm
    best_ms = _calc_makespan(processing_times, best_perm)
    
    # 2. Iterated Local Search (ILS)
    # Unlike Simulated Annealing, ILS uses a deterministic local search (Or-opt)
    # followed by a perturbation to escape local optima. This often explores
    # the search space more effectively by focusing on deep local optima.
    
    T = best_ms * 0.1  # Initial temperature for acceptance criterion
    T_min = 1e-6
    cooling_rate = 0.999
    
    # Or-opt parameters
    # We will try to move blocks of size 1, 2, or 3
    max_block_size = 3
    
    # Perturbation parameters
    # Perturb by swapping two random jobs or moving a block
    perturbation_intensity = 2
    
    for iteration in range(50000):  # Max iterations
        if time.time() > deadline - 2.0:
            break
            
        # --- Local Search: Or-opt ---
        # We perform a first-improvement or best-improvement pass.
        # Given time constraints, we do a few passes of first-improvement.
        improved = True
        passes = 0
        while improved and passes < 3:
            improved = False
            passes += 1
            
            # Shuffle the order of moves to avoid bias? 
            # For speed, we just iterate through all possible moves.
            # To speed up, we can randomly sample a subset of moves if N is large.
            # But for N=100, N^2 = 10000 moves. 
            # 10000 * 50 (M) = 500,000 ops. 
            # If we do 100 passes, that's 50M ops. Might be too slow in Python.
            # Let's stick to a single pass of best-improvement or first-improvement per iteration.
            
            # We will do a first-improvement pass over all possible Or-opt moves.
            # If a move improves, we take it and restart the scan (or continue).
            # Restarting is better for convergence but slower.
            
            for i in range(n_jobs):
                for b_size in range(1, max_block_size + 1):
                    if i + b_size > n_jobs:
                        continue
                    
                    block = current_perm[i:i+b_size]
                    
                    # Skip if block is at the beginning and we're moving it to the beginning (no-op)
                    # Or if it's at the end and moving to the end
                    
                    # Remove block
                    temp_perm = current_perm[:i] + current_perm[i+b_size:]
                    
                    # Try inserting at all other positions
                    # We can limit the search range to nearby positions for speed?
                    # No, Or-opt usually scans all.
                    # But to save time, we can scan a subset or just do first-improvement.
                    
                    # Optimization: We can calculate the delta more efficiently?
                    # Not easily without complex bookkeeping. 
                    # Let's just evaluate.
                    
                    # To speed up, we can skip positions that are "close" to the original if we know it won't help?
                    # Hard to say.
                    
                    # Let's just do a first-improvement scan.
                    # We'll iterate through insertion positions.
                    # To avoid checking the same position (or adjacent which might be equivalent in some cases),
                    # we just check all valid positions in temp_perm.
                    
                    found_improvement = False
                    for k in range(len(temp_perm) + 1):
                        # Skip if k corresponds to the original position (roughly)
                        # Original position was i. In temp_perm, the index i is where the block was removed.
                        # If we insert at i, it goes back. If we insert at i+1, it goes back (since we removed b_size items, index i+1 is now where i+b_size was).
                        # Actually, inserting at i or i+1 in the reduced list restores the original order.
                        # So we can skip k == i and k == i+1 (if i+1 <= len(temp_perm)).
                        if k == i or k == i + 1:
                            continue
                            
                        new_perm = temp_perm[:k] + block + temp_perm[k:]
                        new_ms = _calc_makespan(processing_times, new_perm)
                        
                        if new_ms < best_ms:  # Global best
                            best_ms = new_ms
                            best_perm = new_perm
                            current_perm = new_perm
                            improved = True
                            found_improvement = True
                            break
                        elif new_ms < _calc_makespan(processing_times, current_perm):
                            # First improvement over current (local best)
                            current_perm = new_perm
                            improved = True
                            found_improvement = True
                            break
                            
                    if found_improvement:
                        # Restart the inner loops since the permutation changed
                        break
                        
                if found_improvement:
                    break
                    
        # --- Perturbation ---
        # If we didn't improve, or even if we did, we perturb to escape local optima.
        # We perturb the current_perm (which is the best found in the local search phase).
        
        # Choose perturbation type
        if random.random() < 0.5:
            # Swap two random jobs
            i = random.randint(0, n_jobs - 1)
            j = random.randint(0, n_jobs - 1)
            if i != j:
                current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
        else:
            # Move a block of size 1 or 2
            b_size = random.choice([1, 2])
            if b_size > n_jobs:
                b_size = n_jobs
            i = random.randint(0, n_jobs - b_size)
            block = current_perm[i:i+b_size]
            temp_perm = current_perm[:i] + current_perm[i+b_size:]
            k = random.randint(0, len(temp_perm))
            current_perm = temp_perm[:k] + block + temp_perm[k:]
            
        current_ms = _calc_makespan(processing_times, current_perm)
        
        # Acceptance criterion: Metropolis-style
        # If current is better than best, it's already updated.
        # If current is worse, accept with prob exp(-delta/T)
        if current_ms > best_ms:
            delta = current_ms - best_ms
            if random.random() < np.exp(-delta / T):
                # Accept, but don't update best
                pass
            else:
                # Reject, revert to best
                current_perm = best_perm
                current_ms = best_ms
        else:
            # Current is better or equal
            best_ms = current_ms
            best_perm = current_perm
            
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
    return best_perm
# EVOLVE-BLOCK-END
