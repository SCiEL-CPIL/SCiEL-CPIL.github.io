"""
Strategy: Constructive Insertion Heuristic + Adjacent Swap Local Search
This approach builds the permutation incrementally, inserting each job at the position
that minimizes the current makespan (or a proxy metric), and then refines it using
a systematic adjacent swap local search.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(perm, processing_times):
    """
    Compute makespan for a given permutation.
    perm: list of job indices
    processing_times: numpy array of shape (n_jobs, n_machines)
    """
    n_machines = processing_times.shape[1]
    if not perm:
        return 0.0
    
    # C[j, m] = completion time of job at position j on machine m
    # We only need the previous row to compute the current row
    prev_row = np.zeros(n_machines)
    
    for job_idx in perm:
        curr_row = np.zeros(n_machines)
        for m in range(n_machines):
            if m == 0:
                # First machine: depends only on previous job on same machine
                curr_row[m] = prev_row[m] + processing_times[job_idx, m]
            else:
                # Subsequent machines: depends on previous job on same machine 
                # AND previous machine for the same job
                curr_row[m] = max(prev_row[m], curr_row[m-1]) + processing_times[job_idx, m]
        prev_row = curr_row
        
    return prev_row[-1]

def _constructive_insertion(processing_times):
    """
    Build the permutation by inserting jobs one by one.
    At each step, try all possible insertion positions for the next job and pick the one
    that yields the minimum makespan.
    This is O(n^2 * m) which is feasible for typical flow-shop sizes (n < 200).
    """
    n_jobs, n_machines = processing_times.shape
    
    # Sort jobs by total processing time descending (LPT) as a heuristic for initial order
    # This helps the constructive phase find a good starting point
    totals = processing_times.sum(axis=1)
    job_order = list(np.argsort(-totals))
    
    current_perm = []
    
    for job in job_order:
        best_cost = float('inf')
        best_pos = 0
        
        # Try inserting 'job' at every possible position in current_perm
        # Optimization: We can often skip checking positions that are obviously suboptimal,
        # but for robustness and simplicity, we check all.
        # To speed up, we can compute the makespan incrementally or just call _makespan.
        # Given n is likely small-medium, calling _makespan is safe.
        
        # Optimization: If current_perm is empty, just add
        if not current_perm:
            current_perm.append(job)
            continue
            
        # Evaluate all insertion positions
        # We can optimize by noting that inserting at the end is often good, 
        # but we must check all to be sure.
        for pos in range(len(current_perm) + 1):
            # Create new permutation
            # Using list insertion is O(n), which is fine for small n
            new_perm = current_perm[:pos] + [job] + current_perm[pos:]
            cost = _makespan(new_perm, processing_times)
            
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        
        # Insert at the best position
        current_perm.insert(best_pos, job)
        
    return current_perm

def _adjacent_swap_local_search(perm, processing_times, deadline):
    """
    Perform local search using adjacent swaps.
    This is a dense neighborhood that helps escape shallow local minima 
    more effectively than random swaps.
    """
    current_perm = list(perm)
    n = len(current_perm)
    current_cost = _makespan(current_perm, processing_times)
    best_perm = list(current_perm)
    best_cost = current_cost
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(n - 1):
            if time.time() > deadline:
                return best_perm, best_cost
                
            # Swap adjacent jobs i and i+1
            new_perm = current_perm.copy()
            new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
            
            new_cost = _makespan(new_perm, processing_times)
            
            if new_cost < current_cost:
                current_perm = new_perm
                current_cost = new_cost
                improved = True
                
                if current_cost < best_cost:
                    best_perm = list(current_perm)
                    best_cost = current_cost
                    
    return best_perm, best_cost

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    """
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # 1. Constructive Phase
    # The constructive insertion heuristic is deterministic and fast.
    # It provides a strong initial solution.
    initial_perm = _constructive_insertion(processing_times)
    initial_cost = _makespan(initial_perm, processing_times)
    
    best_perm = initial_perm
    best_cost = initial_cost
    
    # 2. Local Search Phase
    # Use adjacent swap local search to refine the solution.
    # This is more effective than random 2-opt for flow-shop problems 
    # as it explores the neighborhood more systematically.
    ls_perm, ls_cost = _adjacent_swap_local_search(
        initial_perm, processing_times, deadline
    )
    
    if ls_cost < best_cost:
        best_perm = ls_perm
        best_cost = ls_cost
        
    # 3. Optional: Second pass with different initial ordering if time permits
    # If we have significant time left, try a second constructive run with a different heuristic
    # e.g., SPT (Shortest Processing Time) or Randomized LPT
    elapsed = time.time() - start_time
    remaining_time = deadline - time.time()
    
    if remaining_time > 5.0:
        # Try SPT based constructive
        # Re-sort jobs by total processing time ascending
        totals = processing_times.sum(axis=1)
        job_order_spt = list(np.argsort(totals))
        
        # Simple constructive for SPT
        current_perm_spt = []
        for job in job_order_spt:
            best_cost_spt = float('inf')
            best_pos_spt = 0
            if not current_perm_spt:
                current_perm_spt.append(job)
                continue
            for pos in range(len(current_perm_spt) + 1):
                new_perm = current_perm_spt[:pos] + [job] + current_perm_spt[pos:]
                cost = _makespan(new_perm, processing_times)
                if cost < best_cost_spt:
                    best_cost_spt = cost
                    best_pos_spt = pos
            current_perm_spt.insert(best_pos_spt, job)
            
        # Local search on SPT result
        ls_perm_spt, ls_cost_spt = _adjacent_swap_local_search(
            current_perm_spt, processing_times, deadline
        )
        
        if ls_cost_spt < best_cost:
            best_perm = ls_perm_spt
            best_cost = ls_cost_spt

    return list(best_perm)
# EVOLVE-BLOCK-END
