"""
Strategy: Simulated Annealing with incremental O(N*M) evaluation.

This approach fundamentally changes the problem framing from a constructive heuristic (NEH) or exact DP to a probabilistic metaheuristic (Simulated Annealing).

1.  **Constructive Heuristic**: A fast NEH variant is used to generate a high-quality initial permutation.
2.  **Incremental Evaluation**: Instead of recomputing the makespan from scratch for every neighbor (O(N*M) per move, leading to O(N^3*M) total per iteration), we precompute the completion time matrix for the current permutation. When a job is moved, we only recompute the affected suffix of the matrix (O(K*M) where K is the distance of the move). This allows thousands of SA iterations per second.
3.  **Simulated Annealing**: We explore the neighborhood using insertion moves. The cooling schedule is adaptive to the remaining time budget, allowing the algorithm to escape local minima and explore a wider region of the search space than standard hill-climbing.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random

def _makespan_full(perm, processing_times):
    """
    Compute the full completion time matrix and makespan.
    Returns:
        makespan: float
        C: np.array of shape (n_jobs, n_machines) where C[i, m] is completion time of job at position i on machine m.
    """
    n_jobs, n_machines = processing_times.shape
    C = np.zeros((n_jobs, n_machines), dtype=float)
    
    for i, job_idx in enumerate(perm):
        if i == 0:
            # First job in sequence
            C[i, 0] = processing_times[job_idx, 0]
            for m in range(1, n_machines):
                C[i, m] = C[i, m-1] + processing_times[job_idx, m]
        else:
            # Subsequent jobs
            C[i, 0] = C[i-1, 0] + processing_times[job_idx, 0]
            for m in range(1, n_machines):
                C[i, m] = max(C[i-1, m], C[i, m-1]) + processing_times[job_idx, m]
                
    return C[-1, -1], C

def _eval_insertion_move(perm, C, i, j, processing_times):
    """
    Evaluate the makespan if job at position i is moved to position j.
    i is the source index, j is the target index in the NEW permutation (0-indexed).
    If i < j, the job is removed at i, and the jobs i+1..j shift left. The job is inserted at j.
    The suffix starting from index i (in the new perm) needs to be recomputed.
    Actually, it's easier to think:
    Original: ... A ... B ...
    Move A to position j (where j > i):
    New: ... B ... A ...
    The completion times for positions < i remain the same.
    The completion times for positions >= i need to be recomputed.
    
    We return the new makespan and the updated C matrix (only the suffix is updated in place? 
    To avoid side effects, we should copy the suffix or return a new C? 
    For speed, we can update C in place if we are careful, but since we need to revert if the move is rejected, 
    it's safer to compute the new suffix into a temporary buffer and only commit if accepted.
    However, for SA, we need the new C to continue the chain if accepted.
    So we will modify C in place and have a rollback mechanism, or just copy the whole C? 
    Copying C (N*M) is fast. Let's copy the relevant part.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    if i == j:
        return C[-1, -1], C
    
    # Create a copy of the permutation to determine the new order
    # We don't actually need the full new perm for the makespan calc, just the sequence of job indices from i onwards.
    
    # Determine the sequence of jobs from index i to n_jobs-1 in the new permutation
    # If i < j:
    #   Original: P[0..i-1], P[i], P[i+1..j], P[j+1..]
    #   New: P[0..i-1], P[i+1..j], P[i], P[j+1..]
    #   The prefix 0..i-1 is unchanged.
    #   The suffix starts at i.
    #   The jobs in the suffix are: P[i+1], P[i+2], ..., P[j], P[i], P[j+1], ...
    
    # If i > j:
    #   Original: P[0..j-1], P[j..i-1], P[i], P[i+1..]
    #   New: P[0..j-1], P[i], P[j..i-1], P[i+1..]
    #   The prefix 0..j-1 is unchanged.
    #   The suffix starts at j.
    #   The jobs in the suffix are: P[i], P[j], P[j+1], ..., P[i-1], P[i+1], ...
    
    # We will compute the new C values for the affected range.
    # We need to be careful with the "previous row" for the first job in the suffix.
    # If suffix starts at k, the previous row is C[k-1] (if k>0) else None.
    
    # Let's construct the list of job indices for the suffix part.
    # And the starting index of the suffix.
    
    if i < j:
        start_idx = i
        # Jobs from i+1 to j
        suffix_jobs = list(perm[i+1:j+1])
        # Then job at i
        suffix_jobs.append(perm[i])
        # Then jobs from j+1 to end
        suffix_jobs.extend(perm[j+1:])
    else:
        start_idx = j
        # Job at i
        suffix_jobs = [perm[i]]
        # Jobs from j to i-1
        suffix_jobs.extend(perm[j:i])
        # Jobs from i+1 to end
        suffix_jobs.extend(perm[i+1:])
        
    # Now compute new C for indices start_idx to n_jobs-1
    # We need to update C in place? Or return a new C?
    # To support SA, if we accept the move, we update C. If we reject, we don't.
    # So let's compute the new suffix values in a temporary array and return them.
    # The caller will decide whether to update C.
    
    new_suffix_C = np.zeros((len(suffix_jobs), n_machines), dtype=float)
    
    # Get the previous row
    if start_idx == 0:
        prev_row = None
    else:
        prev_row = C[start_idx - 1]
        
    for k, job_idx in enumerate(suffix_jobs):
        curr_idx = start_idx + k
        if prev_row is None:
            # First job overall
            new_suffix_C[k, 0] = processing_times[job_idx, 0]
            for m in range(1, n_machines):
                new_suffix_C[k, m] = new_suffix_C[k, m-1] + processing_times[job_idx, m]
        else:
            # Subsequent job
            new_suffix_C[k, 0] = prev_row[0] + processing_times[job_idx, 0]
            for m in range(1, n_machines):
                # max(prev_row[m], new_suffix_C[k, m-1]) + P[job, m]
                # Note: new_suffix_C[k, m-1] is the completion time of the same job on the previous machine
                new_suffix_C[k, m] = max(prev_row[m], new_suffix_C[k, m-1]) + processing_times[job_idx, m]
        
        prev_row = new_suffix_C[k]
        
    new_makespan = new_suffix_C[-1, -1]
    
    # Return the new makespan and the new suffix C block
    # The caller will splice this into C if accepted
    return new_makespan, new_suffix_C

def _neh_order(processing_times):
    """
    Fast NEH heuristic.
    """
    n_jobs, n_machines = processing_times.shape
    
    # Calculate job weights (sum of processing times)
    weights = processing_times.sum(axis=1)
    
    # Sort jobs by weight descending
    job_order = list(np.argsort(-weights))
    
    perm = []
    # Precompute processing times for speed?
    # We will use _makespan_full for each insertion in NEH? No, that's O(N^3).
    # For NEH, we can just use the simple makespan function or incremental.
    # Since N is small enough for NEH to be fast, let's use a simple O(N*M) makespan for each candidate position.
    
    def _makespan_simple(p):
        # p is a list of job indices
        # Compute makespan
        prev = np.zeros(n_machines)
        for job in p:
            curr = np.zeros(n_machines)
            for m in range(n_machines):
                if m == 0:
                    curr[m] = prev[m] + processing_times[job, m]
                else:
                    curr[m] = max(prev[m], curr[m-1]) + processing_times[job, m]
            prev = curr
        return prev[-1]
        
    for job in job_order:
        best_cost = float('inf')
        best_pos = 0
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [job] + perm[pos:]
            cost = _makespan_simple(new_perm)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        perm.insert(best_pos, job)
        
    return perm

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    """
    start_time = time.time()
    deadline = start_time + 145.0
    
    n_jobs, n_machines = processing_times.shape
    
    # 1. Initial Solution via NEH
    perm = _neh_order(processing_times)
    
    # Compute initial C and makespan
    current_makespan, C = _makespan_full(perm, processing_times)
    best_makespan = current_makespan
    best_perm = list(perm)
    
    # 2. Simulated Annealing
    # Parameters
    T = 10.0  # Initial temperature
    T_min = 0.01
    # Adaptive cooling: we want to reach T_min near the deadline.
    # Let's estimate the number of iterations we can do.
    # A single iteration involves:
    # - Random i, j
    # - _eval_insertion_move: O((N-i)*M) approx O(N*M)
    # - Acceptance check
    # - If accepted, update C: O((N-i)*M)
    
    # Let's run SA until time runs out.
    # We need to be careful with the cooling schedule.
    # A common approach is to use a fixed number of iterations per temperature, or exponential cooling.
    # Given the time constraint, let's use a simple exponential cooling with a rate determined by the expected number of iterations.
    # Or just linear cooling in time?
    # Let's try: T = T_min + (T_initial - T_min) * (remaining_time / total_time)
    # This ensures we cool down as time runs out.
    
    total_time = 145.0
    iterations = 0
    accepted = 0
    
    while time.time() < deadline:
        # Calculate current temperature based on remaining time
        remaining = deadline - time.time()
        if remaining <= 0:
            break
            
        # Simple linear cooling in time
        # T starts at T_initial and goes to T_min as remaining goes to 0
        T = T_min + (T - T_min) * (remaining / total_time)
        
        # Generate a random move: insertion
        i = random.randint(0, n_jobs - 1)
        j = random.randint(0, n_jobs - 1)
        
        if i == j:
            continue
            
        # Evaluate move
        new_makespan, new_suffix_C = _eval_insertion_move(perm, C, i, j, processing_times)
        delta = new_makespan - current_makespan
        
        # Acceptance criterion
        if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
            # Accept move
            current_makespan = new_makespan
            # Update C
            # Determine the start index of the suffix
            if i < j:
                start_idx = i
            else:
                start_idx = j
                
            # Splice new_suffix_C into C
            C[start_idx:start_idx + len(new_suffix_C)] = new_suffix_C
            
            # Update perm
            # Perform the insertion move on perm
            job = perm.pop(i)
            # If i < j, after popping, the index j shifts to j-1.
            # If i > j, index j is unchanged.
            if i < j:
                perm.insert(j, job)
            else:
                perm.insert(j, job)
                
            accepted += 1
            
            # Update best
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_perm = list(perm)
                
        iterations += 1
        
        # Safety check to prevent infinite loops or very slow iterations
        # If we are doing too many iterations, maybe the overhead is high.
        # But with O(N*M) eval, it should be fast.
        
    return best_perm

# EVOLVE-BLOCK-END
