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

def _insertion_local_search(perm, pt, max_iters=200):
    """
    Insertion-based local search with tabu-like memory to avoid cycling.
    Returns improved permutation and makespan.
    """
    n = len(perm)
    if n <= 1:
        return list(perm), _makespan_fast(perm, pt)
    
    current_perm = list(perm)
    current_ms = _makespan_fast(current_perm, pt)
    
    # Track recently tried moves to avoid immediate cycling
    tabu_moves = set()
    tabu_tenure = 10
    
    improved = True
    iters = 0
    while improved and iters < max_iters:
        improved = False
        iters += 1
        for i in range(n):
            job = current_perm[i]
            # Remove job from position i
            without_job = current_perm[:i] + current_perm[i+1:]
            # Try inserting at each position
            for k in range(n - 1):
                if k == i:
                    continue
                
                move_key = (i, k)
                if move_key in tabu_moves:
                    continue
                
                trial = without_job[:k] + [job] + without_job[k:]
                ms = _makespan_fast(trial, pt)
                if ms < current_ms:
                    current_perm = trial
                    current_ms = ms
                    improved = True
                    # Add to tabu
                    tabu_moves.add(move_key)
                    if len(tabu_moves) > tabu_tenure:
                        # Remove oldest (simple: clear half)
                        tabu_moves.clear()
                    break
            if improved:
                break
    
    return current_perm, current_ms

def _or_opt_move(perm, pt, rng, n_moves=10):
    """
    OR-opt move: remove a block of 1-3 consecutive jobs and reinsert elsewhere.
    Returns list of (new_perm, new_ms) for the moves tried.
    """
    n = len(perm)
    if n <= 3:
        return []
    
    results = []
    for _ in range(n_moves):
        # Choose block length 1, 2, or 3
        block_len = rng.randint(1, min(3, n - 1))
        # Choose start position
        i = rng.randint(0, n - block_len)
        
        # Extract block
        block = perm[i:i+block_len]
        
        # Remove block
        remaining = perm[:i] + perm[i+block_len:]
        
        # Choose insertion position in remaining
        j = rng.randint(0, len(remaining))
        
        # Insert block
        new_perm = remaining[:j] + block + remaining[j:]
        
        if new_perm != perm:
            new_ms = _makespan_fast(new_perm, pt)
            results.append((new_perm, new_ms))
    
    return results

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin for 120s limit
    
    n, m = processing_times.shape
    
    if n <= 1:
        return list(range(n))
    
    # 1. Initial Solution: NEH with insertion
    best_perm, best_ms = _neh(processing_times)
    
    # Apply insertion local search to the NEH solution
    best_perm, best_ms = _insertion_local_search(best_perm, processing_times, max_iters=150)
    
    # 2. Multi-restart with diverse perturbation strategies
    rng = np.random.RandomState(42)
    
    max_restarts = 8
    for restart in range(max_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Create diverse starting point using different perturbation strategies
        current_perm = list(best_perm)
        
        if restart % 3 == 0:
            # Swap perturbation
            num_swaps = max(1, n // 8)
            for _ in range(num_swaps):
                i, j = rng.choice(n, 2, replace=False)
                current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
        elif restart % 3 == 1:
            # OR-opt perturbation: remove and reinsert a block
            block_len = min(3, n // 3)
            i = rng.randint(0, n - block_len)
            block = current_perm[i:i+block_len]
            remaining = current_perm[:i] + current_perm[i+block_len:]
            j = rng.randint(0, len(remaining))
            current_perm = remaining[:j] + block + remaining[j:]
        else:
            # Reverse a segment
            i = rng.randint(0, n - 2)
            j = rng.randint(i + 1, n - 1)
            current_perm[i:j+1] = current_perm[i:j+1][::-1]
        
        current_ms = _makespan_fast(current_perm, processing_times)
        
        # Simulated Annealing with adaptive parameters
        T = max(best_ms * 0.03, 1.0)
        T_min = best_ms * 0.0005
        alpha = 0.9998
        
        # Inner loop: do SA steps until time runs out or T drops too low
        sa_steps = 0
        while time.time() < deadline - 15.0 and T > T_min:
            # Generate neighbor: mix of swaps and OR-opt moves
            if rng.random() < 0.7:
                # Swap move
                i, j = rng.choice(n, 2, replace=False)
                new_perm = current_perm[:]
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            else:
                # OR-opt move
                or_opt_results = _or_opt_move(current_perm, processing_times, rng, n_moves=1)
                if not or_opt_results:
                    T *= alpha
                    continue
                new_perm, new_ms_candidate = or_opt_results[0]
            
            new_ms = _makespan_fast(new_perm, processing_times)
            delta = new_ms - current_ms
            
            if delta < 0 or rng.random() < np.exp(-delta / T):
                current_perm = new_perm
                current_ms = new_ms
                
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_perm = current_perm[:]
            
            T *= alpha
            sa_steps += 1
            
            # Periodically apply quick local search if we've made progress
            if sa_steps % 500 == 0 and current_ms < best_ms * 1.02:
                quick_perm, quick_ms = _insertion_local_search(current_perm, processing_times, max_iters=20)
                if quick_ms < best_ms:
                    best_ms = quick_ms
                    best_perm = quick_perm[:]
                    current_perm = quick_perm
                    current_ms = quick_ms
        
        # Apply insertion local search to the SA result if it's reasonably close
        if current_ms <= best_ms * 1.08:
            polished_perm, polished_ms = _insertion_local_search(current_perm, processing_times, max_iters=80)
            if polished_ms < best_ms:
                best_ms = polished_ms
                best_perm = polished_perm
    
    # Final polish with more iterations if time allows
    if time.time() < deadline - 3.0:
        polished_perm, polished_ms = _insertion_local_search(best_perm, processing_times, max_iters=100)
        if polished_ms < best_ms:
            best_ms = polished_ms
            best_perm = polished_perm
    
    # One more final polish if there's still time
    if time.time() < deadline - 1.0:
        final_perm, final_ms = _insertion_local_search(best_perm, processing_times, max_iters=50)
        if final_ms < best_ms:
            best_ms = final_ms
            best_perm = final_perm
    
    return list(best_perm)
# EVOLVE-BLOCK-END
