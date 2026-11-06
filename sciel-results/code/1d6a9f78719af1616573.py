"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import random


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    C = np.zeros((n_jobs, n_machines))
    for j in range(n_jobs):
        job = perm[j]
        pt = processing_times[job]
        if j == 0:
            C[j, 0] = pt[0]
        else:
            C[j, 0] = C[j-1, 0] + pt[0]
        for m in range(1, n_machines):
            C[j, m] = max(C[j-1, m], C[j, m-1]) + pt[m]
    return float(C[-1, -1])


def _compute_completion_matrix(perm, processing_times):
    """Compute the full completion time matrix C."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    C = np.zeros((n_jobs, n_machines))
    if n_jobs == 0:
        return C
    for j in range(n_jobs):
        job = perm[j]
        pt = processing_times[job]
        if j == 0:
            C[j, 0] = pt[0]
        else:
            C[j, 0] = C[j-1, 0] + pt[0]
        for m in range(1, n_machines):
            C[j, m] = max(C[j-1, m], C[j, m-1]) + pt[m]
    return C


def _compute_suffix_matrix(perm, processing_times, start_idx):
    """
    Compute suffix completion vectors: for each position i from start_idx to n-1,
    compute what the completion vector would be if we started from position i
    with the job at position i, given the completion vector at position i-1.
    
    Actually, let's think differently. For a swap at position i (swapping perm[i] and perm[i+1]),
    we need to know:
    - C_prev = completion vector at position i-1 (or zeros if i==0)
    - The new completion vectors for positions i, i+1, i+2, ...
    
    For the suffix, we can precompute: for each position k >= i+2, the "transfer function"
    from the completion vector at position k-1 to position k. But this is not simply additive
    because of the max() operation.
    
    Alternative approach: For a swap at position i, the completion vectors from i+2 onward
    depend on the new C[i+1]. We can precompute for each position k the "residual" effect.
    
    Actually, the cleanest fast approach: 
    - Precompute C (full completion matrix)
    - For a swap at position i, compute new C[i] and C[i+1] in O(M)
    - Then propagate from i+2 to end in O((N-i-2)*M)
    
    To make this faster, we can precompute a "suffix propagation" structure.
    
    Let's use a different strategy: precompute for each position i, the completion vector
    assuming the prefix up to i-1 is fixed. Then for a swap, we only need to recompute
    from i onward.
    
    For now, let's use the O((N-i)*M) propagation but with early termination if the
    makespan is already worse than the current best.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    # This function will be used in the SA to quickly evaluate swaps
    pass


def _fast_swap_makespan(perm, C, i, processing_times, current_best_ms):
    """
    Fast evaluation of makespan after swapping perm[i] and perm[i+1].
    Uses precomputed C to avoid full recomputation.
    Returns the new makespan, or inf if it's already known to be worse than current_best_ms.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    # Get the completion vector before position i
    if i == 0:
        C_prev = np.zeros(n_machines)
    else:
        C_prev = C[i-1]
    
    job_a = perm[i]
    job_b = perm[i+1]
    pt_a = processing_times[job_a]
    pt_b = processing_times[job_b]
    
    # Compute new C[i] (job_b comes first)
    C_new_i = np.empty(n_machines)
    C_new_i[0] = C_prev[0] + pt_b[0]
    for m in range(1, n_machines):
        C_new_i[m] = max(C_prev[m], C_new_i[m-1]) + pt_b[m]
    
    # Compute new C[i+1] (job_a comes second)
    C_new_i1 = np.empty(n_machines)
    C_new_i1[0] = C_new_i[0] + pt_a[0]
    for m in range(1, n_machines):
        C_new_i1[m] = max(C_new_i[m], C_new_i1[m-1]) + pt_a[m]
    
    # Early termination: if the last machine's completion at i+1 already exceeds current_best_ms
    # and all remaining jobs add positive time, we can return early
    if C_new_i1[-1] >= current_best_ms:
        return float('inf')
    
    # If i+1 is the last position, we're done
    if i + 1 == n_jobs - 1:
        return float(C_new_i1[-1])
    
    # Propagate from i+2 to end
    C_curr = C_new_i1
    for j in range(i + 2, n_jobs):
        job = perm[j]
        pt = processing_times[job]
        C_next = np.empty(n_machines)
        C_next[0] = C_curr[0] + pt[0]
        for m in range(1, n_machines):
            C_next[m] = max(C_curr[m], C_next[m-1]) + pt[m]
        C_curr = C_next
        # Early termination
        if C_curr[-1] >= current_best_ms:
            return float('inf')
    
    return float(C_curr[-1])


def _sa_run(perm, processing_times, start_time, deadline, rng, n_jobs, n_machines):
    """
    Single simulated annealing run with fast incremental evaluation.
    """
    current_perm = list(perm)
    C = _compute_completion_matrix(current_perm, processing_times)
    current_ms = float(C[-1, -1])
    best_perm = list(current_perm)
    best_ms = current_ms
    
    # Temperature schedule
    initial_temp = max(current_ms * 0.1, 1.0)
    final_temp = max(current_ms * 0.0005, 0.01)
    cooling_rate = 0.998
    
    temp = initial_temp
    steps = 0
    
    while time.time() < deadline:
        if temp < final_temp:
            temp = final_temp
            break
        
        # Generate neighbor: adjacent swap
        i = rng.randint(0, n_jobs - 2)
        
        # Fast evaluation
        new_ms = _fast_swap_makespan(current_perm, C, i, processing_times, best_ms)
        
        delta = new_ms - current_ms if new_ms < float('inf') else current_ms * 0.1
        
        # Acceptance criterion
        accept = False
        if new_ms < float('inf') and (new_ms < current_ms or rng.random() < math.exp(-delta / temp)):
            accept = True
        
        if accept:
            # Apply the swap
            current_perm[i], current_perm[i+1] = current_perm[i+1], current_perm[i]
            
            # Update C incrementally
            C_new_i = np.empty(n_machines)
            C_prev = np.zeros(n_machines) if i == 0 else C[i-1]
            job_b = current_perm[i]
            job_a = current_perm[i+1]
            pt_b = processing_times[job_b]
            pt_a = processing_times[job_a]
            
            C_new_i[0] = C_prev[0] + pt_b[0]
            for m in range(1, n_machines):
                C_new_i[m] = max(C_prev[m], C_new_i[m-1]) + pt_b[m]
            
            C_new_i1 = np.empty(n_machines)
            C_new_i1[0] = C_new_i[0] + pt_a[0]
            for m in range(1, n_machines):
                C_new_i1[m] = max(C_new_i[m], C_new_i1[m-1]) + pt_a[m]
            
            C[i] = C_new_i
            C[i+1] = C_new_i1
            
            # Propagate the rest
            C_curr = C_new_i1
            for j in range(i + 2, n_jobs):
                job = current_perm[j]
                pt = processing_times[job]
                C_next = np.empty(n_machines)
                C_next[0] = C_curr[0] + pt[0]
                for m in range(1, n_machines):
                    C_next[m] = max(C_curr[m], C_next[m-1]) + pt[m]
                C[j] = C_next
                C_curr = C_next
            
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = list(current_perm)
        
        temp *= cooling_rate
        steps += 1
    
    return best_perm, best_ms


def _local_search(perm, processing_times, max_iterations=50, deadline=None):
    """
    Local search with adjacent swaps using fast evaluation.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs <= 1:
        return list(perm)
    
    best_perm = list(perm)
    C = _compute_completion_matrix(best_perm, processing_times)
    best_ms = float(C[-1, -1])
    
    for it in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        for i in range(n_jobs - 1):
            ms = _fast_swap_makespan(best_perm, C, i, processing_times, best_ms)
            if ms < best_ms - 1e-10:
                # Apply the swap
                best_perm[i], best_perm[i+1] = best_perm[i+1], best_perm[i]
                best_ms = ms
                
                # Update C
                C_prev = np.zeros(n_machines) if i == 0 else C[i-1]
                job_b = best_perm[i]
                job_a = best_perm[i+1]
                pt_b = processing_times[job_b]
                pt_a = processing_times[job_a]
                
                C_new_i = np.empty(n_machines)
                C_new_i[0] = C_prev[0] + pt_b[0]
                for m in range(1, n_machines):
                    C_new_i[m] = max(C_prev[m], C_new_i[m-1]) + pt_b[m]
                
                C_new_i1 = np.empty(n_machines)
                C_new_i1[0] = C_new_i[0] + pt_a[0]
                for m in range(1, n_machines):
                    C_new_i1[m] = max(C_new_i[m], C_new_i1[m-1]) + pt_a[m]
                
                C[i] = C_new_i
                C[i+1] = C_new_i1
                
                C_curr = C_new_i1
                for j in range(i + 2, n_jobs):
                    job = best_perm[j]
                    pt = processing_times[job]
                    C_next = np.empty(n_machines)
                    C_next[0] = C_curr[0] + pt[0]
                    for m in range(1, n_machines):
                        C_next[m] = max(C_curr[m], C_next[m-1]) + pt[m]
                    C[j] = C_next
                    C_curr = C_next
                
                improved = True
                break
        
        if not improved:
            break
    
    return best_perm


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Generate multiple random starting permutations
    n_restarts = max(3, min(8, int((deadline - start_time) / 15.0)))
    
    best_perm_overall = None
    best_ms_overall = float('inf')
    
    # Time per restart
    time_per_restart = (deadline - start_time - 5.0) / n_restarts
    
    for r in range(n_restarts):
        if time.time() > deadline - 5.0:
            break
        
        # Generate a random starting permutation
        rng = random.Random(42 + r * 1000)
        perm = list(range(n_jobs))
        rng.shuffle(perm)
        
        # Run SA
        sa_deadline = start_time + (r + 1) * time_per_restart
        if sa_deadline > deadline - 5.0:
            sa_deadline = deadline - 5.0
        
        sa_perm, sa_ms = _sa_run(perm, processing_times, start_time, sa_deadline, rng, n_jobs, n_machines)
        
        if sa_ms < best_ms_overall:
            best_ms_overall = sa_ms
            best_perm_overall = sa_perm
    
    # Final local search polish
    if best_perm_overall is not None and time.time() < deadline - 2.0:
        polished = _local_search(best_perm_overall, processing_times, max_iterations=100, deadline=deadline)
        polished_ms = _makespan(polished, processing_times)
        if polished_ms < best_ms_overall:
            best_perm_overall = polished
            best_ms_overall = polished_ms
    
    return list(best_perm_overall)
# EVOLVE-BLOCK-END
