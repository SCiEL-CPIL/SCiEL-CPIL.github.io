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
    Insertion-based local search: try moving each job to every other position.
    Returns improved permutation and makespan.
    """
    n = len(perm)
    if n <= 1:
        return list(perm), _makespan_fast(perm, pt)
    
    current_perm = list(perm)
    current_ms = _makespan_fast(current_perm, pt)
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n):
            job = current_perm[i]
            without_job = current_perm[:i] + current_perm[i+1:]
            for k in range(n - 1):
                if k == i:
                    continue
                trial = without_job[:k] + [job] + without_job[k:]
                ms = _makespan_fast(trial, pt)
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
    tabu_list = []
    
    # Temperature for accepting worse moves (aspiration-like)
    T = best_ms * 0.02
    T_min = best_ms * 0.0005
    alpha = 0.999
    
    start_time = time.time()
    stall_count = 0
    max_stall = 500
    
    while time.time() < time_limit:
        if T < T_min:
            break
        
        # Generate a neighbor
        new_perm, move_key = _or_opt_move(current_perm, n)
        new_ms = _makespan_fast(new_perm, pt)
        
        # Check tabu status
        is_tabu = move_key in tabu_list
        
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
            tabu_list.append(move_key)
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
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
    
    max_restarts = 8
    for restart in range(max_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Perturb the best solution to create a diverse starting point
        current_perm = list(best_perm)
        
        # Use or-opt perturbation for more effective diversification
        num_perturbs = max(2, n // 5)
        for _ in range(num_perturbs):
            if n > 1:
                perturbed_perm, _ = _or_opt_move(current_perm, n)
                current_perm = perturbed_perm
        
        current_ms = _makespan_fast(current_perm, processing_times)
        
        # Run tabu search
        tabu_start = time.time()
        tabu_time_limit = deadline - 20.0
        ts_perm, ts_ms = _tabu_search(current_perm, processing_times, rng, tabu_time_limit)
        
        if ts_ms < best_ms:
            best_ms = ts_ms
            best_perm = ts_perm
        
        # Apply insertion local search to the tabu search result if it's reasonably close
        if current_ms <= best_ms * 1.1 and time.time() < deadline - 10.0:
            polished_perm, polished_ms = _insertion_local_search(current_perm, processing_times, max_iters=80)
            if polished_ms < best_ms:
                best_ms = polished_ms
                best_perm = polished_perm
        
        # Also polish the tabu result
        if time.time() < deadline - 10.0:
            polished_perm, polished_ms = _insertion_local_search(ts_perm, processing_times, max_iters=80)
            if polished_ms < best_ms:
                best_ms = polished_ms
                best_perm = polished_perm
    
    # Final polish with more iterations if time allows
    if time.time() < deadline - 3.0:
        polished_perm, polished_ms = _insertion_local_search(best_perm, processing_times, max_iters=150)
        if polished_ms < best_ms:
            best_ms = polished_ms
            best_perm = polished_perm
    
    return list(best_perm)
# EVOLVE-BLOCK-END
