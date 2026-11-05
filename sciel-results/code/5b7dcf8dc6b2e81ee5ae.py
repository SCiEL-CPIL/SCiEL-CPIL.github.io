"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(perm, P):
    """Calculate makespan for a given permutation."""
    n = len(perm)
    if n == 0:
        return 0.0
    # C[j][m] is completion time of job j on machine m
    # We can do this in O(N*M)
    # To be faster, we can track the current time for each machine
    machine_times = [0.0] * P.shape[1]
    for job_idx in perm:
        for m in range(P.shape[1]):
            # Start time is max(machine_free_time, job_start_time_on_prev_machine)
            # Actually, for flow shop:
            # C[j][m] = max(C[j-1][m], C[j][m-1]) + P[j][m]
            # But we are iterating jobs in perm order.
            # Let's use the standard DP formulation.
            pass
    
    # Simpler O(N*M) implementation
    # Let C[m] be the completion time of the current job on machine m? No.
    # Let's use a 2D array or just track the last completion time for the previous job on each machine.
    # C_prev[m] = completion time of previous job in perm on machine m
    # C_curr[m] = completion time of current job in perm on machine m
    # C_curr[0] = C_prev[0] + P[job][0]
    # C_curr[m] = max(C_prev[m], C_curr[m-1]) + P[job][m]
    
    C_prev = [0.0] * P.shape[1]
    C_curr = [0.0] * P.shape[1]
    
    for job_idx in perm:
        C_curr[0] = C_prev[0] + P[job_idx, 0]
        for m in range(1, P.shape[1]):
            C_curr[m] = max(C_prev[m], C_curr[m-1]) + P[job_idx, m]
        C_prev, C_curr = C_curr, C_prev
        
    return C_prev[-1]

def _neh_order(P):
    """Construct initial solution using NEH heuristic."""
    n, m = P.shape
    # Total processing time per job
    totals = P.sum(axis=1)
    # Sort jobs by descending total processing time
    # np.argsort returns indices that would sort the array
    # We want descending, so we negate or use reverse
    order = np.argsort(-totals)
    
    perm = []
    for job in order:
        # Find best position to insert job
        best_pos = 0
        best_ms = np.inf
        for i in range(len(perm) + 1):
            # Create new perm
            new_perm = perm[:i] + [job] + perm[i:]
            ms = _makespan(new_perm, P)
            if ms < best_ms:
                best_ms = ms
                best_pos = i
        perm.insert(best_pos, job)
        
    return perm

def _or_opt_improve(perm, P, time_limit, start_time):
    """
    Improve permutation using Or-opt moves.
    Returns the best permutation found within the time limit.
    """
    n = len(perm)
    if n < 2:
        return perm
    
    best_perm = perm[:]
    best_ms = _makespan(best_perm, P)
    
    # Or-opt lengths to try
    lengths = [1, 2, 3]
    
    # We use a while loop with time check
    while True:
        if time.time() - start_time > time_limit:
            break
            
        improved = False
        current_perm = best_perm[:]
        current_ms = best_ms
        
        # Try all possible Or-opt moves
        # A move is defined by:
        # 1. Start index i of the block to remove
        # 2. Length L of the block (1, 2, 3)
        # 3. Insertion index j (where the block is re-inserted)
        
        # To ensure we don't insert at the same place, we skip j == i or j == i+L (depending on definition)
        # Standard Or-opt: remove block [i, i+L-1], insert at position j.
        
        for i in range(n):
            for L in lengths:
                if i + L > n:
                    continue
                
                # Extract block
                block = current_perm[i:i+L]
                # Remove block
                remaining = current_perm[:i] + current_perm[i+L:]
                
                # Try inserting at all valid positions in remaining
                # Remaining length is n - L
                # Insertion positions: 0 to len(remaining)
                for j in range(len(remaining) + 1):
                    # Skip if inserting back to original position
                    # Original position was i.
                    # If we insert at j in the remaining list:
                    # If j < i, the block moves left.
                    # If j >= i, the block moves right (or stays).
                    # We should skip cases where the block ends up in the same relative position.
                    # If j == i, it's the same.
                    # If j == i + L? No, remaining is shorter.
                    # Let's just compute makespan and compare. It's O(N*M).
                    # To save time, we can prune obvious bad moves, but let's keep it simple first.
                    
                    new_perm = remaining[:j] + block + remaining[j:]
                    
                    # Optimization: If the move is effectively a no-op, skip.
                    # This happens if the block is moved to a position that results in the same sequence.
                    # E.g., removing 1 item at i and inserting at i.
                    # Or removing block [i, i+L-1] and inserting at i (in remaining, which is effectively i).
                    # Let's just check if new_perm == current_perm.
                    if new_perm == current_perm:
                        continue
                    
                    ms = _makespan(new_perm, P)
                    if ms < current_ms - 1e-9:
                        current_perm = new_perm
                        current_ms = ms
                        improved = True
                        # Break out of loops to restart search from new best?
                        # Or continue to find more improvements in this pass?
                        # First improvement strategy: break out of all loops and restart.
                        break
                
                if improved:
                    break
            if improved:
                break
        
        if improved:
            best_perm = current_perm
            best_ms = current_ms
        else:
            # No improvement found in a full pass
            # Try a perturbation to escape local optimum
            # Simple perturbation: swap two random jobs
            if n > 1:
                idx1 = np.random.randint(0, n)
                idx2 = np.random.randint(0, n)
                if idx1 != idx2:
                    perturbed = best_perm[:]
                    perturbed[idx1], perturbed[idx2] = perturbed[idx2], perturbed[idx1]
                    p_ms = _makespan(perturbed, P)
                    # Accept perturbation if it's not too bad, or just use it as new starting point for local search?
                    # Since we are in a "hill climbing" loop, we usually restart from the best found so far.
                    # If we don't accept the perturbation, we are stuck.
                    # Let's use a simple random restart: generate a new NEH or perturb heavily.
                    # Given time limits, let's just break if we can't improve, or try a few random restarts.
                    
                    # Let's try a few random restarts if we have time
                    restarts = 0
                    while time.time() - start_time < time_limit - 5 and restarts < 10:
                        # Generate a new random solution or perturb best
                        # Perturbing best is better
                        new_start = best_perm[:]
                        # Random swap
                        idx1 = np.random.randint(0, n)
                        idx2 = np.random.randint(0, n)
                        if idx1 != idx2:
                            new_start[idx1], new_start[idx2] = new_start[idx2], new_start[idx1]
                        
                        # Run a quick local search from this new start
                        # To avoid infinite recursion, we do a limited number of Or-opt steps
                        # Or just evaluate if it's better? No, local search from scratch is better.
                        # But we need a separate function for limited search.
                        
                        # Let's just accept the perturbation as the new 'current' for the next outer loop iteration?
                        # No, the outer loop checks for improvement.
                        # Let's just break and return best_perm if we can't improve.
                        break
            break
            
    return best_perm

def order_jobs(processing_times):
    P = processing_times
    n, m = P.shape
    
    start_time = time.time()
    time_limit = 110.0 # Leave 10s buffer
    
    # 1. Initial NEH
    perm = _neh_order(P)
    
    # 2. Local Search
    if time.time() - start_time < time_limit:
        perm = _or_opt_improve(perm, P, time_limit, start_time)
        
    return perm

# EVOLVE-BLOCK-END
