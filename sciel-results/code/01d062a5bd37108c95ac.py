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

def _insertion_local_search(perm, pt, max_iters=200, deadline=None):
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
        if deadline is not None and time.time() > deadline:
            break
        improved = False
        iters += 1
        for i in range(n):
            if deadline is not None and time.time() > deadline:
                break
            job = current_perm[i]
            # Remove job from position i
            without_job = current_perm[:i] + current_perm[i+1:]
            # Try inserting at each position
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

def _tabu_search(perm, pt, tabu_tenure=7, max_iters=500, deadline=None):
    """
    Tabu search with insertion moves.
    Uses a tabu list of (job, position) pairs to prevent cycling.
    Aspiration criterion: accept tabu move if it improves global best.
    """
    n = len(perm)
    if n <= 1:
        return list(perm), _makespan_fast(perm, pt)
    
    current_perm = list(perm)
    current_ms = _makespan_fast(current_perm, pt)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    # Tabu list: set of (job_idx, target_position)
    # We store the move that was made: (job, from_pos, to_pos)
    # To untabu, we check if the reverse move is in the tabu list
    # Simpler: store (job, pos) meaning "job should not be moved TO pos"
    # But that's not quite right for swaps/insertions.
    # Let's store the actual move: (job, from, to)
    # A move is tabu if (job, from, to) is in the tabu set
    # We check the reverse move (job, to, from) to see if it's tabu
    
    tabu_list = []  # list of (job, from_pos, to_pos)
    tabu_set = set()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        best_move = None
        best_move_ms = float('inf')
        best_move_is_tabu = False
        
        # Generate all possible insertion moves
        for i in range(n):
            job = current_perm[i]
            without_job = current_perm[:i] + current_perm[i+1:]
            for k in range(n - 1):
                if k == i:
                    continue
                # Check if this move is tabu
                # The move is (job, i, k)
                # It is tabu if the reverse move (job, k, i) was made recently
                # But our tabu list stores (job, from, to)
                # So we check if (job, k, i) is in tabu_set
                is_tabu = (job, k, i) in tabu_set
                
                if is_tabu:
                    continue
                    
                trial = without_job[:k] + [job] + without_job[k:]
                ms = _makespan_fast(trial, pt)
                
                if ms < best_move_ms:
                    best_move_ms = ms
                    best_move = (job, i, k, trial)
                    best_move_is_tabu = False
        
        # If no non-tabu move found, accept the best tabu move (aspiration)
        if best_move is None:
            # Find best move regardless of tabu status
            for i in range(n):
                job = current_perm[i]
                without_job = current_perm[:i] + current_perm[i+1:]
                for k in range(n - 1):
                    if k == i:
                        continue
                    trial = without_job[:k] + [job] + without_job[k:]
                    ms = _makespan_fast(trial, pt)
                    if ms < best_move_ms:
                        best_move_ms = ms
                        best_move = (job, i, k, trial)
                        best_move_is_tabu = True
        
        if best_move is None:
            break
            
        job, i, k, new_perm = best_move
        
        # Update tabu list
        tabu_list.append((job, i, k))
        tabu_set.add((job, i, k))
        
        if len(tabu_list) > tabu_tenure:
            old_move = tabu_list.pop(0)
            tabu_set.remove(old_move)
        
        current_perm = new_perm
        current_ms = best_move_ms
        
        if current_ms < best_ms:
            best_ms = current_ms
            best_perm = current_perm[:]
            
    return best_perm, best_ms

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s safety margin for 120s limit
    
    n, m = processing_times.shape
    
    if n <= 1:
        return list(range(n))
    
    # 1. Initial Solution: NEH with insertion
    best_perm, best_ms = _neh(processing_times)
    
    # Apply insertion local search to the NEH solution
    best_perm, best_ms = _insertion_local_search(best_perm, processing_times, max_iters=100, deadline=deadline - 5.0)
    
    # 2. Multi-restart Tabu Search
    rng = np.random.RandomState(42)
    
    max_restarts = 5
    for restart in range(max_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Perturb the best solution to create a diverse starting point
        current_perm = list(best_perm)
        num_swaps = max(1, n // 10)
        for _ in range(num_swaps):
            i, j = rng.choice(n, 2, replace=False)
            current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
        
        current_ms = _makespan_fast(current_perm, processing_times)
        
        # Tabu Search
        # Adjust tabu tenure based on problem size
        tabu_tenure = max(5, min(15, n // 5))
        ts_perm, ts_ms = _tabu_search(current_perm, processing_times, 
                                       tabu_tenure=tabu_tenure, 
                                       max_iters=300, 
                                       deadline=deadline - 5.0)
        
        if ts_ms < best_ms:
            best_ms = ts_ms
            best_perm = ts_perm
    
    # Final polish with insertion local search if time allows
    if time.time() < deadline - 2.0:
        polished_perm, polished_ms = _insertion_local_search(best_perm, processing_times, max_iters=100, deadline=deadline - 1.0)
        if polished_ms < best_ms:
            best_ms = polished_ms
            best_perm = polished_perm
    
    return list(best_perm)
# EVOLVE-BLOCK-END
