"""Flow-shop makespan minimizer: NEH + Fast Local Search with Delta Evaluation."""
# EVOLVE-BLOCK-START
import time
import numpy as np


def _calc_makespan(perm, pt):
    """
    Calculate the makespan for a given permutation.
    perm: array of job indices
    pt: processing times (n_jobs, n_machines)
    Returns: makespan (float)
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    if n_jobs == 0:
        return 0.0
    
    p = pt[perm]
    c = np.zeros(n_jobs, dtype=float)
    
    for m in range(n_machines):
        for j in range(n_jobs):
            if j == 0:
                start_time = c[j]
            else:
                start_time = max(c[j], c[j-1])
            c[j] = start_time + p[j, m]
            
    return c[-1]


def _completion_matrix(perm, pt):
    """
    Compute the full completion time matrix C[j, m] = completion time of job at position j on machine m.
    Returns C of shape (n_jobs, n_machines).
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    if n_jobs == 0:
        return np.empty((0, n_machines))
    
    p = pt[perm]
    C = np.zeros((n_jobs, n_machines), dtype=float)
    
    for m in range(n_machines):
        for j in range(n_jobs):
            if j == 0:
                C[j, m] = (C[j, m-1] if m > 0 else 0.0) + p[j, m]
            else:
                C[j, m] = max(C[j, m-1] if m > 0 else 0.0, C[j-1, m]) + p[j, m]
    
    return C


def _makespan_from_C(C):
    """Extract makespan from completion matrix."""
    if C.size == 0:
        return 0.0
    return C[-1, -1]


def _delta_2opt(C, pt, i, j):
    """
    Compute delta makespan for reversing segment [i, j] (inclusive).
    Only the sub-matrix C[i:j+1, :] changes. We recompute that portion.
    Returns new makespan (not delta) for efficiency in comparison.
    """
    n_jobs = C.shape[0]
    n_machines = C.shape[1]
    
    # The jobs affected are positions i through j
    # We need to reverse the segment and recompute completions for those positions
    # and all subsequent positions
    
    # Create a copy of the relevant portion
    # Jobs before i are unchanged
    # Jobs from i to j are reversed
    # Jobs after j may change
    
    # Get the processing times for the segment
    # perm[i:j+1] reversed -> we need the actual job indices
    # But we only have C, not perm. So this approach won't work directly.
    # We need perm as well.
    
    return None  # Placeholder - we'll use a different approach


def _eval_2opt(perm, pt, C, i, j):
    """
    Evaluate the makespan after reversing segment [i, j] in perm.
    Uses the completion matrix C for fast recomputation.
    Returns new makespan.
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    
    # Create new perm with segment reversed
    # We only need to recompute C from position i onwards
    # Jobs 0..i-1 are unchanged, so C[0:i, :] stays the same
    
    # New segment: perm[i:j+1] reversed
    new_segment = perm[i:j+1][::-1]
    new_perm_segment = new_segment
    
    # Recompute C for positions i onwards
    # For position i, the "previous job on same machine" is C[i-1, m] (if i>0)
    # and the "previous machine completion" for the job at position i is C[i, m-1] (if m>0)
    # But wait - the job at position i in the new perm is different, so C[i, m-1] changes too.
    
    # Actually, we need to recompute the entire tail from position i.
    # Let's build a new C for positions i..n-1
    
    # The job at new position k (k >= i) is:
    # If k is in [i, j]: new job = perm[i+j-k] (reversed)
    # If k > j: new job = perm[k] (unchanged)
    
    # We need to recompute C[k, m] for all k >= i, m >= 0
    # C[k, m] depends on C[k, m-1] (same job, prev machine) and C[k-1, m] (prev job, same machine)
    
    # Let's use a temporary array for the new C values
    new_C = C.copy()
    
    # Process positions from i to n_jobs-1
    for k in range(i, n_jobs):
        # Determine which job is at position k in the new perm
        if k <= j:
            job_idx = perm[i + j - k]
        else:
            job_idx = perm[k]
        
        p_job = pt[job_idx]
        
        for m in range(n_machines):
            # C[k, m] = max(C[k, m-1], C[k-1, m]) + p_job[m]
            # where C[k, m-1] is from the new_C (already computed for this k, m-1)
            # and C[k-1, m] is from the new_C (already computed for position k-1)
            
            prev_machine = new_C[k, m-1] if m > 0 else 0.0
            prev_job = new_C[k-1, m] if k > 0 else 0.0
            
            new_C[k, m] = max(prev_machine, prev_job) + p_job[m]
    
    return new_C[-1, -1]


def _eval_swap(perm, pt, C, i, j):
    """
    Evaluate the makespan after swapping positions i and j in perm.
    Only positions i and j change, but all subsequent positions may be affected.
    Returns new makespan.
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    
    # Swap jobs at positions i and j
    # New job at position i: perm[j]
    # New job at position j: perm[i]
    # All other positions unchanged
    
    new_C = C.copy()
    
    # Recompute from position i onwards
    for k in range(i, n_jobs):
        # Determine which job is at position k in the new perm
        if k == i:
            job_idx = perm[j]
        elif k == j:
            job_idx = perm[i]
        else:
            job_idx = perm[k]
        
        p_job = pt[job_idx]
        
        for m in range(n_machines):
            prev_machine = new_C[k, m-1] if m > 0 else 0.0
            prev_job = new_C[k-1, m] if k > 0 else 0.0
            
            new_C[k, m] = max(prev_machine, prev_job) + p_job[m]
    
    return new_C[-1, -1]


def _eval_insert(perm, pt, C, i, j):
    """
    Evaluate the makespan after removing job at position i and inserting at position j.
    If i < j: job moves from i to j, jobs i+1..j shift left by 1
    If i > j: job moves from i to j, jobs j..i-1 shift right by 1
    Returns new makespan.
    """
    n_jobs = len(perm)
    n_machines = pt.shape[1]
    
    if i == j:
        return _makespan_from_C(C)
    
    new_C = C.copy()
    
    # Determine the starting position for recomputation
    if i < j:
        # Job at i is removed, jobs i+1..j shift left, job goes to position j
        # Positions affected: i, i+1, ..., j
        start_pos = i
    else:
        # Job at i is removed, jobs j..i-1 shift right, job goes to position j
        # Positions affected: j, j+1, ..., i
        start_pos = j
    
    # Build the new job sequence for positions start_pos onwards
    # Original: perm[0], perm[1], ..., perm[n-1]
    # Remove perm[i], insert at position j
    
    # For each position k >= start_pos, determine which job is there
    for k in range(start_pos, n_jobs):
        if i < j:
            # Job at original position i is moved to position j
            # Positions i..j-1 in new perm correspond to original positions i+1..j
            # Position j in new perm is original position i
            if k < j:
                job_idx = perm[k + 1]  # shifted left
            elif k == j:
                job_idx = perm[i]      # the moved job
            else:
                job_idx = perm[k]      # unchanged
        else:
            # i > j
            # Job at original position i is moved to position j
            # Positions j..i-1 in new perm correspond to original positions j..i-1 shifted right
            # Position j in new perm is original position i
            if k == j:
                job_idx = perm[i]      # the moved job
            elif k > j and k <= i:
                job_idx = perm[k - 1]  # shifted right
            else:
                job_idx = perm[k]      # unchanged
        
        p_job = pt[job_idx]
        
        for m in range(n_machines):
            prev_machine = new_C[k, m-1] if m > 0 else 0.0
            prev_job = new_C[k-1, m] if k > 0 else 0.0
            
            new_C[k, m] = max(prev_machine, prev_job) + p_job[m]
    
    return new_C[-1, -1]


def _neh(processing_times):
    """
    NEH heuristic: order jobs by total processing time desc, insert each optimally.
    """
    n_jobs, n_machines = processing_times.shape
    order = list(np.argsort(-processing_times.sum(axis=1)))
    
    perm = np.empty(n_jobs, dtype=np.int64)
    perm[0] = order[0]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_cost = float('inf')
        
        for pos in range(i + 1):
            cand = np.empty(i + 1, dtype=np.int64)
            if pos == 0:
                cand[0] = job
                cand[1:] = perm[:i]
            elif pos == i:
                cand[:i] = perm[:i]
                cand[i] = job
            else:
                cand[:pos] = perm[:pos]
                cand[pos] = job
                cand[pos + 1:] = perm[pos:i]
            
            cost = _calc_makespan(cand, processing_times)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        
        new_perm = np.empty(i + 1, dtype=np.int64)
        if best_pos == 0:
            new_perm[0] = job
            new_perm[1:] = perm[:i]
        elif best_pos == i:
            new_perm[:i] = perm[:i]
            new_perm[i] = job
        else:
            new_perm[:best_pos] = perm[:best_pos]
            new_perm[best_pos] = job
            new_perm[best_pos + 1:] = perm[best_pos:i]
        
        perm[:i + 1] = new_perm
        
    return perm


def _local_search(perm, processing_times, deadline, margin):
    """
    Local search using 2-opt, swap, and insertion moves with fast delta evaluation.
    """
    n_jobs, n_machines = processing_times.shape
    best = perm.copy()
    C = _completion_matrix(best, processing_times)
    best_cost = _makespan_from_C(C)
    improved = True
    
    while improved:
        improved = False
        
        # 2-Opt: Reverse a segment [i, j]
        for i in range(n_jobs - 1):
            if time.monotonic() > deadline - margin:
                return best, best_cost
            for j in range(i + 1, n_jobs):
                new_cost = _eval_2opt(best, processing_times, C, i, j)
                if new_cost < best_cost - 1e-12:
                    # Apply the move
                    best[i:j + 1] = best[i:j + 1][::-1]
                    C = _completion_matrix(best, processing_times)
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Swap: Exchange two positions i, j
        for i in range(n_jobs):
            if time.monotonic() > deadline - margin:
                return best, best_cost
            for j in range(i + 1, n_jobs):
                new_cost = _eval_swap(best, processing_times, C, i, j)
                if new_cost < best_cost - 1e-12:
                    best[i], best[j] = best[j], best[i]
                    C = _completion_matrix(best, processing_times)
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break
        
        if improved:
            continue
        
        # Insertion: Remove job at i, insert at j
        for i in range(n_jobs):
            if time.monotonic() > deadline - margin:
                return best, best_cost
            for j in range(n_jobs):
                if i == j:
                    continue
                new_cost = _eval_insert(best, processing_times, C, i, j)
                if new_cost < best_cost - 1e-12:
                    # Apply insertion
                    job = best[i]
                    if i < j:
                        best[i:j] = best[i+1:j+1]
                        best[j] = job
                    else:
                        best[j+1:i+1] = best[j:i]
                        best[j] = job
                    C = _completion_matrix(best, processing_times)
                    best_cost = new_cost
                    improved = True
                    break
            if improved:
                break
                
    return best, best_cost


def order_jobs(processing_times):
    processing_times = np.asarray(processing_times, dtype=float)
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]

    t0 = time.monotonic()
    deadline = t0 + 145.0  # Safety margin
    margin = 5.0

    # 1. Constructive Heuristic: NEH
    best_perm = _neh(processing_times)
    best_cost = _calc_makespan(best_perm, processing_times)
    
    # 2. Local Search with fast evaluation
    best_perm, best_cost = _local_search(best_perm, processing_times, deadline, margin)
    
    # 3. If time remains, try alternative heuristics
    # SPT on first machine
    spt_perm = np.argsort(processing_times[:, 0])
    spt_cost = _calc_makespan(spt_perm, processing_times)
    if spt_cost < best_cost:
        best_perm = spt_perm
        best_cost = spt_cost
        best_perm, best_cost = _local_search(best_perm, processing_times, deadline, margin)
    
    # Random restarts
    rng = np.random.default_rng(42)
    for _ in range(10):
        if time.monotonic() > deadline - margin:
            break
        rand_perm = rng.permutation(n_jobs)
        rand_cost = _calc_makespan(rand_perm, processing_times)
        if rand_cost < best_cost:
            best_perm = rand_perm
            best_cost = rand_cost
            best_perm, best_cost = _local_search(best_perm, processing_times, deadline, margin)
            
    return list(best_perm)
# EVOLVE-BLOCK-END
