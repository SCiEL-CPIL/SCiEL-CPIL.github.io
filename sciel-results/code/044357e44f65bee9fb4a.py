"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math


def _makespan_fast(processing_times, order):
    """
    Compute makespan for a given job order using efficient recurrence.
    Uses the standard flow-shop recurrence:
    C[i][j] = max(C[i-1][j], C[i][j-1]) + P[i][j]
    where C[-1][j] = 0, C[i][-1] = 0
    """
    if len(order) == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    
    # c[j] holds the completion time of the current job on machine j
    # We process jobs one by one, updating c in-place
    c = [0.0] * n_machines
    for job_idx in order:
        p_row = processing_times[job_idx]
        for j in range(n_machines):
            if j == 0:
                c[j] = c[j] + p_row[j]
            else:
                c[j] = max(c[j], c[j-1]) + p_row[j]
    return c[-1]


def _makespan_fast_np(processing_times, order):
    """
    NumPy-accelerated makespan computation.
    For larger instances, this can be significantly faster.
    """
    if len(order) == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    
    # Extract processing times for jobs in order
    # Shape: (n_jobs, n_machines)
    p_matrix = processing_times[order]
    n_jobs = len(order)
    
    # C[i][j] = max(C[i-1][j], C[i][j-1]) + P[i][j]
    # We can compute this row by row
    c = np.zeros(n_machines)
    for i in range(n_jobs):
        p_row = p_matrix[i]
        # c[j] = max(c[j], c[j-1]) + p_row[j]
        # Need to be careful: c[j-1] on the right is the NEW c[j-1] (already updated)
        # and c[j] on the right is the OLD c[j] (from previous job)
        for j in range(n_machines):
            if j == 0:
                c[j] = c[j] + p_row[j]
            else:
                c[j] = max(c[j], c[j-1]) + p_row[j]
    return c[-1]


def _neh_heuristic(processing_times, sort_key='sum'):
    """
    NEH (Nawaz, Enscore, Ham) heuristic for flow shop scheduling.
    Jobs are sorted by a key, then each is inserted at the position
    that minimizes the makespan of the partial sequence.
    """
    n_jobs, n_machines = processing_times.shape
    
    if sort_key == 'sum':
        totals = processing_times.sum(axis=1)
    elif sort_key == 'mean':
        totals = processing_times.mean(axis=1)
    elif sort_key == 'min':
        totals = processing_times.min(axis=1)
    elif sort_key == 'max':
        totals = processing_times.max(axis=1)
    elif sort_key == 'range':
        totals = processing_times.max(axis=1) - processing_times.min(axis=1)
    elif sort_key == 'variance':
        totals = processing_times.var(axis=1)
    else:
        totals = processing_times.sum(axis=1)
    
    # Sort jobs by the key in descending order
    sorted_jobs = list(np.argsort(-totals))
    
    order = []
    for job in sorted_jobs:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(order) + 1):
            candidate = order[:pos] + [job] + order[pos:]
            ms = _makespan_fast(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        order.insert(best_pos, job)
    
    return order


def _local_search_insertion(processing_times, order, time_limit=5.0, max_passes=50):
    """
    First-improvement local search using insertion moves.
    Repeatedly removes each job and reinserts it at the best position.
    """
    n_jobs = len(order)
    if n_jobs <= 1:
        return order, _makespan_fast(processing_times, order)
    
    current_order = list(order)
    current_ms = _makespan_fast(processing_times, current_order)
    start_time = time.time()
    
    for _ in range(max_passes):
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            
            job = current_order[i]
            # Remove job from position i
            remaining = current_order[:i] + current_order[i+1:]
            
            # Find best insertion position
            best_pos = i
            best_ms = current_ms
            
            for pos in range(len(remaining) + 1):
                if pos == i:
                    continue
                candidate = remaining[:pos] + [job] + remaining[pos:]
                ms = _makespan_fast(processing_times, candidate)
                if ms < best_ms:
                    best_ms = ms
                    best_pos = pos
            
            if best_ms < current_ms:
                current_order = remaining[:best_pos] + [job] + remaining[best_pos:]
                current_ms = best_ms
                improved = True
        
        if not improved:
            break
    
    return current_order, current_ms


def _simulated_annealing(processing_times, order, time_limit=10.0):
    """
    Simulated annealing with insertion moves to escape local optima.
    """
    n_jobs = len(order)
    if n_jobs <= 1:
        return order, _makespan_fast(processing_times, order)
    
    current_order = list(order)
    current_ms = _makespan_fast(processing_times, current_order)
    
    best_order = current_order[:]
    best_ms = current_ms
    
    start_time = time.time()
    initial_temp = max(current_ms * 0.02, 1.0)
    cooling_rate = 0.9995
    temp = initial_temp
    
    while time.time() - start_time < time_limit and temp > 0.01:
        # Generate random insertion move
        src = random.randint(0, n_jobs - 1)
        dst = random.randint(0, n_jobs - 1)
        if dst == src or dst == src + 1:
            temp *= cooling_rate
            continue
        
        # Apply move
        job = current_order[src]
        remaining = current_order[:src] + current_order[src+1:]
        # Adjust dst index after removal
        if src < dst:
            dst -= 1
        new_order = remaining[:dst] + [job] + remaining[dst:]
        
        new_ms = _makespan_fast(processing_times, new_order)
        delta = new_ms - current_ms
        
        # Metropolis acceptance criterion
        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_order = new_order
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_order = current_order[:]
        
        temp *= cooling_rate
    
    return best_order, best_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on flow-shop instances.
    
    Strategy:
    1. Generate multiple diverse initial solutions using NEH with different sorting keys
    2. Apply local search to each
    3. Use simulated annealing on the best to escape local optima
    4. Final local search polish
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs <= 1:
        return list(range(n_jobs))
    
    # Phase 1: Generate diverse initial solutions using NEH variants
    sort_keys = ['sum', 'mean', 'max', 'min', 'range']
    candidates = []
    
    for key in sort_keys:
        if time.time() > deadline - 30.0:
            break
        try:
            order = _neh_heuristic(processing_times, sort_key=key)
            ms = _makespan_fast(processing_times, order)
            candidates.append((ms, order))
        except Exception:
            continue
    
    # Also add a random initial solution
    if time.time() < deadline - 30.0:
        random_order = list(range(n_jobs))
        random.shuffle(random_order)
        ms = _makespan_fast(processing_times, random_order)
        candidates.append((ms, random_order))
    
    if not candidates:
        # Fallback: identity permutation
        return list(range(n_jobs))
    
    # Sort by makespan
    candidates.sort(key=lambda x: x[0])
    
    # Phase 2: Local search on top candidates
    ls_time_per_candidate = 3.0
    best_order = candidates[0][1]
    best_ms = candidates[0][0]
    
    for ms, order in candidates[:min(3, len(candidates))]:
        if time.time() > deadline - 40.0:
            break
        ls_order, ls_ms = _local_search_insertion(
            processing_times, order, 
            time_limit=ls_time_per_candidate
        )
        if ls_ms < best_ms:
            best_ms = ls_ms
            best_order = ls_order
    
    # Phase 3: Simulated annealing on the best solution
    remaining_time = deadline - time.time()
    if remaining_time > 15.0:
        sa_time = min(remaining_time * 0.6, 50.0)
        sa_order, sa_ms = _simulated_annealing(
            processing_times, best_order,
            time_limit=sa_time
        )
        if sa_ms < best_ms:
            best_ms = sa_ms
            best_order = sa_order
    
    # Phase 4: Final local search polish
    remaining_time = deadline - time.time()
    if remaining_time > 5.0:
        polish_time = min(remaining_time * 0.5, 15.0)
        final_order, final_ms = _local_search_insertion(
            processing_times, best_order,
            time_limit=polish_time
        )
        if final_ms < best_ms:
            best_ms = final_ms
            best_order = final_order
    
    return best_order
# EVOLVE-BLOCK-END
