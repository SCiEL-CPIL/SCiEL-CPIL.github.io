"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan_fast(perm, pt):
    """Faster makespan computation using list-based perm."""
    if len(perm) == 0:
        return 0.0
    m = pt.shape[1]
    c = [0.0] * m
    for job_idx in perm:
        prev_c = 0.0
        row = pt[job_idx]
        for k in range(m):
            val = row[k]
            if c[k] > prev_c:
                c[k] = c[k] + val
            else:
                c[k] = prev_c + val
            prev_c = c[k]
    return c[-1]

def _completion_matrix(perm, pt):
    """Compute full completion time matrix C[i][k]."""
    n = len(perm)
    m = pt.shape[1]
    C = [[0.0] * m for _ in range(n)]
    if n == 0:
        return C
    for i in range(n):
        job = perm[i]
        row = pt[job]
        for k in range(m):
            if k == 0:
                C[i][k] = (C[i-1][k] if i > 0 else 0.0) + row[0]
            else:
                C[i][k] = max(C[i-1][k], C[i][k-1]) + row[k]
    return C

def _makespan_from_C(perm, pt, C, start_idx):
    """
    Given completion matrix C for perm, recompute makespan from start_idx onward
    assuming perm[start_idx:] is unchanged but prefix completion times are given.
    C[start_idx-1] gives completion times after position start_idx-1.
    Returns (new_C_suffix, makespan).
    """
    n = len(perm)
    m = pt.shape[1]
    if start_idx >= n:
        return C, C[n-1][m-1] if n > 0 else 0.0
    
    new_C = [row[:] for row in C]
    for idx in range(start_idx, n):
        job = perm[idx]
        row = pt[job]
        for k in range(m):
            if k == 0:
                if idx == 0:
                    new_C[idx][k] = row[0]
                else:
                    new_C[idx][k] = new_C[idx-1][k] + row[0]
            else:
                new_C[idx][k] = max(new_C[idx-1][k], new_C[idx][k-1]) + row[k]
    return new_C, new_C[n-1][m-1]

def _compute_completion(perm, pt):
    """Compute completion times C[j][k] for each job j on machine k."""
    n = len(perm)
    m = pt.shape[1]
    C = [[0.0] * m for _ in range(n)]
    if n == 0:
        return C
    for i, job_idx in enumerate(perm):
        row = pt[job_idx]
        for k in range(m):
            if k == 0:
                C[i][k] = C[i-1][k] + row[0] if i > 0 else row[0]
            else:
                C[i][k] = max(C[i-1][k], C[i][k-1]) + row[k]
    return C

def _insertion_delta(perm, pt, i, k):
    """
    Compute delta makespan for moving job from position i to position k.
    Returns the new makespan without recomputing everything.
    """
    n = len(perm)
    m = pt.shape[1]
    if i == k or i == k + 1:
        return None  # no change
    
    # Build completion matrix for current perm
    C = _compute_completion(perm, pt)
    current_ms = C[n-1][m-1]
    
    job = perm[i]
    row = pt[job]
    
    # Remove job from position i
    new_perm = perm[:i] + perm[i+1:]
    n2 = n - 1
    
    # Recompute C for new_perm (without job)
    C2 = [[0.0] * m for _ in range(n2)]
    for idx, j in enumerate(new_perm):
        r = pt[j]
        for k2 in range(m):
            if k2 == 0:
                C2[idx][k2] = C2[idx-1][k2] + r[0] if idx > 0 else r[0]
            else:
                C2[idx][k2] = max(C2[idx-1][k2], C2[idx][k2-1]) + r[k2]
    
    # Insert job at position k in new_perm
    # Compute completion times up to position k-1 in new_perm
    # Then insert job, then compute the rest
    
    # For positions before k, use C2
    # At position k, the job is inserted
    # After k, recompute
    
    # Completion of job at position k
    if k == 0:
        c_job = [row[0]] + [0.0] * (m-1)
        for k2 in range(1, m):
            c_job[k2] = c_job[k2-1] + row[k2]
    else:
        c_job = [0.0] * m
        for k2 in range(m):
            if k2 == 0:
                c_job[k2] = C2[k-1][k2] + row[0]
            else:
                c_job[k2] = max(C2[k-1][k2], c_job[k2-1]) + row[k2]
    
    # Now compute completion times for jobs after position k
    # These are new_perm[k:] shifted to positions k+1, k+2, ...
    new_ms = max(c_job[m-1], 0.0)
    for idx in range(k, n2):
        j = new_perm[idx]
        r = pt[j]
        for k2 in range(m):
            if k2 == 0:
                if idx == k:
                    c_val = c_job[0] + r[0]
                else:
                    c_val = C_after[idx-1][0] + r[0]
            else:
                if idx == k:
                    c_val = max(c_job[k2-1] if k2 > 0 else 0, C_after[idx-1][k2] if idx > k else 0) + r[k2]
                else:
                    c_val = max(C_after[idx-1][k2], C_after[idx][k2-1]) + r[k2]
            # This is getting complicated, let's just recompute from k onward
    
    # Simpler: just recompute from position k onward
    C3 = [[0.0] * m for _ in range(n)]
    for idx in range(n):
        if idx < k:
            C3[idx] = C2[idx][:]
        elif idx == k:
            C3[idx] = c_job[:]
        else:
            j = new_perm[idx-1]  # new_perm has n-1 elements, position idx-1 maps to old position idx
            r = pt[j]
            for k2 in range(m):
                if k2 == 0:
                    C3[idx][k2] = C3[idx-1][k2] + r[0]
                else:
                    C3[idx][k2] = max(C3[idx-1][k2], C3[idx][k2-1]) + r[k2]
    
    return C3[n-1][m-1]

def _neh(processing_times):
    """NEH heuristic: build solution by inserting jobs in order of decreasing total processing time."""
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n)), 0.0
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    perm = []
    for job in order:
        best_pos = -1
        best_ms = float('inf')
        for k in range(len(perm) + 1):
            trial = perm[:k] + [job] + perm[k:]
            ms = _makespan_fast(trial, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = k
        perm.insert(best_pos, job)
    
    return perm, _makespan_fast(perm, processing_times)

def _insertion_local_search(perm, pt, max_iters=500):
    """
    Insertion-based local search with incremental makespan evaluation.
    Returns improved permutation and makespan.
    """
    n = len(perm)
    if n <= 1:
        return list(perm), _makespan_fast(perm, pt)
    
    current_perm = list(perm)
    current_ms = _makespan_fast(current_perm, pt)
    m = pt.shape[1]
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        
        # Compute completion matrix for current perm
        C = _completion_matrix(current_perm, pt)
        
        for i in range(n):
            job = current_perm[i]
            row = pt[job]
            without_job = current_perm[:i] + current_perm[i+1:]
            n2 = n - 1
            
            for k in range(n2 + 1):
                if k == i or k == i + 1:
                    continue
                
                # Build trial permutation
                trial = without_job[:k] + [job] + without_job[k:]
                
                # Fast evaluation: reuse C for prefix up to min(i,k)-1
                # The affected region is from min(i,k) to n-1
                start = min(i, k)
                
                # Compute completion from start onward
                new_C = [row[:] for row in C]  # copy
                for idx in range(start, n):
                    j = trial[idx]
                    r = pt[j]
                    for kk in range(m):
                        if kk == 0:
                            if idx == 0:
                                new_C[idx][kk] = r[0]
                            else:
                                new_C[idx][kk] = new_C[idx-1][kk] + r[0]
                        else:
                            if idx == 0:
                                new_C[idx][kk] = new_C[idx][kk-1] + r[kk]
                            else:
                                new_C[idx][kk] = max(new_C[idx-1][kk], new_C[idx][kk-1]) + r[kk]
                
                ms = new_C[n-1][m-1]
                if ms < current_ms:
                    current_perm = trial
                    current_ms = ms
                    improved = True
                    break
            if improved:
                break
    
    return current_perm, current_ms

def _or_opt_move(perm, n):
    """
    Generate an or-opt move: remove a segment of length 1, 2, or 3 and reinsert elsewhere.
    Returns (new_perm, move_key) where move_key is a tuple for tabu list.
    """
    # Choose segment length: 1 (50%), 2 (30%), 3 (20%)
    r = random.random()
    if r < 0.5:
        seg_len = 1
    elif r < 0.8:
        seg_len = 2
    else:
        seg_len = 3
    
    if n <= seg_len:
        seg_len = max(1, n - 1)
    
    # Choose start position for segment
    start = random.randint(0, n - seg_len)
    
    # Choose insertion position (not the same as original)
    insert_pos = random.randint(0, n - seg_len)
    while insert_pos == start:
        insert_pos = random.randint(0, n - seg_len)
    
    # Extract segment
    segment = perm[start:start + seg_len]
    
    # Build new permutation
    without_segment = perm[:start] + perm[start + seg_len:]
    
    # Adjust insertion position if it was after the removed segment
    if insert_pos > start:
        insert_pos -= seg_len
    
    new_perm = without_segment[:insert_pos] + segment + without_segment[insert_pos:]
    
    # Move key for tabu: (start, insert_pos, seg_len)
    move_key = (start, insert_pos, seg_len)
    
    return new_perm, move_key

def _tabu_search(perm, pt, rng, time_limit):
    """
    Tabu search with or-opt moves.
    Returns best permutation and makespan found.
    """
    n = len(perm)
    if n <= 1:
        return list(perm), _makespan_fast(perm, pt)
    
    current_perm = list(perm)
    current_ms = _makespan_fast(current_perm, pt)
    best_perm = list(current_perm)
    best_ms = current_ms
    
    # Tabu list: store move keys for tabu_tenure iterations
    tabu_tenure = max(10, n // 2)
    from collections import deque
    tabu_queue = deque()
    tabu_set = set()
    
    # Temperature for accepting worse moves (aspiration-like)
    T = best_ms * 0.02
    T_min = best_ms * 0.0005
    alpha = 0.9995
    
    start_time = time.time()
    stall_count = 0
    max_stall = 300
    
    while time.time() < time_limit:
        if T < T_min:
            break
        
        # Generate a neighbor
        new_perm, move_key = _or_opt_move(current_perm, n)
        new_ms = _makespan_fast(new_perm, pt)
        
        # Check tabu status - O(1) with set
        is_tabu = move_key in tabu_set
        
        # Aspiration: accept if it improves best
        aspiration = new_ms < best_ms
        
        if new_ms < current_ms or (not is_tabu and (aspiration or rng.random() < np.exp(-(new_ms - current_ms) / T))):
            # Accept the move
            current_perm = new_perm
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = list(current_perm)
                stall_count = 0
            else:
                stall_count += 1
            
            # Update tabu list
            tabu_queue.append(move_key)
            tabu_set.add(move_key)
            if len(tabu_queue) > tabu_tenure:
                old_key = tabu_queue.popleft()
                tabu_set.discard(old_key)
        else:
            stall_count += 1
        
        T *= alpha
        
        if stall_count > max_stall:
            break
    
    return best_perm, best_ms

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 150s limit
    
    n, m = processing_times.shape
    
    if n <= 1:
        return list(range(n))
    
    # 1. Initial Solution: NEH with insertion
    best_perm, best_ms = _neh(processing_times)
    
    # Apply insertion local search to the NEH solution
    best_perm, best_ms = _insertion_local_search(best_perm, processing_times, max_iters=100)
    
    # 2. Multi-restart Tabu Search with or-opt moves
    rng = np.random.RandomState(42)
    
    max_restarts = 10
    for restart in range(max_restarts):
        if time.time() > deadline - 20.0:
            break
        
        # Perturb the best solution to create a diverse starting point
        current_perm = list(best_perm)
        
        # Use or-opt perturbation for more effective diversification
        num_perturbs = max(3, n // 4)
        for _ in range(num_perturbs):
            if n > 1:
                perturbed_perm, _ = _or_opt_move(current_perm, n)
                current_perm = perturbed_perm
        
        current_ms = _makespan_fast(current_perm, processing_times)
        
        # Run tabu search with time budget
        remaining = deadline - 25.0 - time.time()
        if remaining < 2.0:
            break
        tabu_time_limit = time.time() + remaining
        ts_perm, ts_ms = _tabu_search(current_perm, processing_times, rng, tabu_time_limit)
        
        if ts_ms < best_ms:
            best_ms = ts_ms
            best_perm = ts_perm
        
        # Polish the tabu result
        if time.time() < deadline - 12.0:
            polished_perm, polished_ms = _insertion_local_search(ts_perm, processing_times, max_iters=100)
            if polished_ms < best_ms:
                best_ms = polished_ms
                best_perm = polished_perm
    
    # Final intensive polish if time allows
    if time.time() < deadline - 5.0:
        remaining = deadline - 5.0 - time.time()
        # Scale iterations based on remaining time
        max_iters = min(300, max(50, int(remaining * 30)))
        polished_perm, polished_ms = _insertion_local_search(best_perm, processing_times, max_iters=max_iters)
        if polished_ms < best_ms:
            best_ms = polished_ms
            best_perm = polished_perm
    
    return list(best_perm)
# EVOLVE-BLOCK-END
