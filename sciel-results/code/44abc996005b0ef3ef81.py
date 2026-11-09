"""Improved program for flow-shop permutation construction using NEH and 2-Opt."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(order, P):
    """
    Compute the makespan for a given job order.
    order: list of job indices
    P: np.ndarray of shape (n_jobs, n_machines)
    """
    if len(order) == 0:
        return 0.0
    
    n_machines = P.shape[1]
    # C[m] is the completion time of the current job on machine m
    # We process jobs in the order given.
    # For a flow shop, the completion time of job j on machine m is:
    # C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m]
    # We only need the final makespan, which is C[last_job, last_machine].
    # However, to compute C[j, m], we need C[j-1, m] and C[j, m-1].
    # We can maintain an array C_prev_machine for the previous job's completion times?
    # No, C[j, m-1] is the completion time of the *current* job on the *previous* machine.
    # C[j-1, m] is the completion time of the *previous* job on the *current* machine.
    
    # Let's use a 1D array to store the completion time of the last processed job on each machine.
    # Actually, we need the completion time of the *previous job* on machine m.
    # Let comp[m] be the completion time of the job previously scheduled on machine m.
    # Initially comp = [0, 0, ..., 0].
    # For each job j in order:
    #   curr_comp = 0
    #   for m in range(n_machines):
    #       curr_comp = max(comp[m], curr_comp) + P[j, m]
    #       comp[m] = curr_comp
    #   # After the loop, comp[m] holds the completion time of job j on machine m.
    #   # The makespan is the final value of curr_comp (which is comp[n_machines-1]).
    
    comp = np.zeros(n_machines)
    makespan = 0.0
    for j in order:
        curr_comp = 0.0
        for m in range(n_machines):
            # The job cannot start on machine m before it finishes on m-1 (curr_comp)
            # AND before machine m is free (comp[m])
            curr_comp = max(comp[m], curr_comp) + P[j, m]
            comp[m] = curr_comp
        makespan = curr_comp
    return makespan

def _neh_heuristic(P):
    """
    Implement the NEH heuristic.
    """
    n_jobs, n_machines = P.shape
    
    # 1. Calculate total processing times
    totals = P.sum(axis=1)
    
    # 2. Sort jobs by decreasing total processing time
    sorted_jobs = np.argsort(-totals)
    
    # 3. Construct permutation
    perm = []
    for j in sorted_jobs:
        best_pos = 0
        best_comp = float('inf')
        
        # Try inserting job j at every position in the current perm
        for pos in range(len(perm) + 1):
            new_perm = perm[:pos] + [j] + perm[pos:]
            
            # Calculate the completion time of job j on the last machine
            # We don't need the full makespan of new_perm, just C[j, M]
            # But C[j, M] depends on the previous jobs in new_perm.
            
            # Optimization: We can simulate the flow shop up to job j.
            # Since j is at position 'pos', it is the (pos)-th job in the sequence.
            # We need the completion times of the jobs before j on all machines.
            
            # Let's just calculate the full makespan of new_perm? 
            # That is O(n*m) per insertion, so O(n^2*m) total. 
            # For n=100, m=10, this is 100,000 operations. Very fast.
            
            # However, we can be smarter.
            # We only care about the completion time of job j on the last machine.
            # Let's simulate the schedule for new_perm up to job j.
            
            comp = np.zeros(n_machines)
            curr_comp = 0.0
            for idx, job_idx in enumerate(new_perm):
                for m in range(n_machines):
                    curr_comp = max(comp[m], curr_comp) + P[job_idx, m]
                    comp[m] = curr_comp
                if idx == pos:
                    # We just finished processing job j
                    # curr_comp is C[j, n_machines-1]
                    if curr_comp < best_comp:
                        best_comp = curr_comp
                        best_pos = pos
                    break
            
        perm = perm[:best_pos] + [j] + perm[best_pos:]
        
    return perm

def _two_opt(perm, P):
    """
    Perform 2-opt local search to improve the permutation.
    """
    n_jobs = len(perm)
    if n_jobs < 2:
        return perm
    
    best_perm = perm
    best_ms = _makespan(best_perm, P)
    
    improved = True
    while improved:
        improved = False
        for i in range(n_jobs - 1):
            for j in range(i + 1, n_jobs):
                # Swap jobs at i and j
                new_perm = best_perm.copy()
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
                
                ms = _makespan(new_perm, P)
                if ms < best_ms:
                    best_ms = ms
                    best_perm = new_perm
                    improved = True
        # If we want to be faster, we can break early, but full pass is safer for quality
        # in the time budget.
        
    return best_perm

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    
    P = np.asarray(processing_times, dtype=float)
    n_jobs, n_machines = P.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Step 1: NEH Heuristic
    perm = _neh_heuristic(P)
    best_perm = perm
    best_ms = _makespan(best_perm, P)
    
    # Step 2: 2-Opt Local Search
    # We run 2-opt until no improvement or time limit
    while True:
        if time.time() > deadline:
            break
        
        new_perm = _two_opt(best_perm, P)
        new_ms = _makespan(new_perm, P)
        
        if new_ms >= best_ms:
            break
        
        best_perm = new_perm
        best_ms = new_ms
        
    # Check time again
    if time.time() > deadline:
        return list(best_perm)
        
    return list(best_perm)
# EVOLVE-BLOCK-END
