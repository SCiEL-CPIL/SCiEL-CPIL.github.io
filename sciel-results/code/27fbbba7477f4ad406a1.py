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
    prev_row = np.zeros(n_machines)
    for j in range(n_jobs):
        job = perm[j]
        pt = processing_times[job]
        cur_row = np.zeros(n_machines)
        cur_row[0] = prev_row[0] + pt[0]
        if n_machines > 1:
            for m in range(1, n_machines):
                cur_row[m] = max(prev_row[m], cur_row[m-1]) + pt[m]
        prev_row = cur_row
    return float(prev_row[-1])


def _makespan_from_prev(perm, processing_times, prev_completion):
    """
    Compute makespan starting from a known completion vector for a prefix.
    prev_completion[m] is the completion time on machine m after some prefix.
    perm is the full permutation, and we compute from scratch but this is
    just the same as _makespan. Keeping for clarity.
    """
    return _makespan(perm, processing_times)


def _insert_best_position(job, partial, processing_times):
    """Insert job into partial list at the position minimizing makespan."""
    n_partial = len(partial)
    best_pos = 0
    best_ms = float('inf')
    
    for pos in range(n_partial + 1):
        candidate = partial[:pos] + [job] + partial[pos:]
        ms = _makespan(candidate, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_pos = pos
    
    return partial[:best_pos] + [job] + partial[best_pos:], best_ms


def _neh_order(processing_times, rng=None):
    """Generate NEH heuristic ordering with optional randomization."""
    if rng is None:
        rng = random.Random()
    
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    
    # Sort by total processing time descending
    order = list(np.argsort(-totals))
    
    # Add slight randomization to break ties and diversify
    if n_jobs > 2:
        # Swap a few adjacent elements randomly
        for _ in range(min(3, n_jobs // 5)):
            i = rng.randint(0, n_jobs - 2)
            if rng.random() < 0.3:
                order[i], order[i+1] = order[i+1], order[i]
    
    partial = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        partial, _ = _insert_best_position(job, partial, processing_times)
    
    return partial


def _neh_order_forward_backward(processing_times, rng=None):
    """NEH with forward and backward passes."""
    if rng is None:
        rng = random.Random()
    
    n_jobs, n_machines = processing_times.shape
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Forward pass
    partial = [order[0]]
    for i in range(1, n_jobs):
        job = order[i]
        partial, _ = _insert_best_position(job, partial, processing_times)
    
    # Backward pass: remove each job and reinsert at best position
    result = list(partial)
    for i in range(n_jobs - 1, 0, -1):
        # Remove job at position i
        job = result[i]
        remaining = result[:i] + result[i+1:]
        result, _ = _insert_best_position(job, remaining, processing_times)
    
    return result


def _swap_adjacent(perm, i):
    """Create new permutation with adjacent swap at position i."""
    new_perm = list(perm)
    new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
    return new_perm


def _insert_job(perm, job, pos):
    """Insert job at position pos in perm (removing it from current position if present)."""
    if job in perm:
        perm = [j for j in perm if j != job]
    return perm[:pos] + [job] + perm[pos:]


def _move_job(perm, job, new_pos):
    """Move job to new position."""
    perm = [j for j in perm if j != job]
    return perm[:new_pos] + [job] + perm[new_pos:]


def _local_search_2opt(perm, processing_times, max_iterations=200, deadline=None):
    """
    Local search with 2-opt (adjacent swap) and insert moves.
    Uses Best-Improvement for swaps and First-Improvement for inserts to balance speed and quality.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm)
    
    best_perm = list(perm)
    best_ms = _makespan(best_perm, processing_times)
    
    for it in range(max_iterations):
        if deadline is not None and time.time() > deadline:
            break
        
        improved = False
        
        # 1. Adjacent Swaps (Best Improvement)
        best_swap_pos = -1
        best_swap_ms = best_ms
        for i in range(n_jobs - 1):
            new_perm = _swap_adjacent(best_perm, i)
            ms = _makespan(new_perm, processing_times)
            if ms < best_swap_ms - 1e-10:
                best_swap_ms = ms
                best_swap_pos = i
        
        if best_swap_pos != -1:
            best_perm = _swap_adjacent(best_perm, best_swap_pos)
            best_ms = best_swap_ms
            improved = True
        
        # 2. Insert Moves (First Improvement)
        # Scan all possible moves, but stop at the first one that improves the solution.
        # This can be significantly faster if improvements are common.
        found_improvement = False
        for i in range(n_jobs):
            job = best_perm[i]
            # Optimization: Only try positions that are significantly different?
            # For now, try all, but break early.
            for pos in range(n_jobs):
                if pos == i or pos == i + 1:
                    continue
                new_perm = _move_job(best_perm, job, pos)
                ms = _makespan(new_perm, processing_times)
                if ms < best_ms - 1e-10:
                    best_perm = new_perm
                    best_ms = ms
                    improved = True
                    found_improvement = True
                    break
            if found_improvement:
                break
        
        if not improved:
            break
    
    return best_perm


def _simulated_annealing(perm, processing_times, start_time, deadline, rng=None):
    """
    Simple simulated annealing with adjacent swap and insert moves.
    """
    if rng is None:
        rng = random.Random()
    
    n_jobs = len(perm)
    if n_jobs <= 1:
        return list(perm)
    
    current_perm = list(perm)
    current_ms = _makespan(current_perm, processing_times)
    best_perm = list(current_perm)
    best_ms = current_ms
    
    # Set initial temperature based on problem size
    initial_temp = current_ms * 0.1
    cooling_rate = 0.999
    
    temp = initial_temp
    
    while time.time() < deadline:
        if temp < 1e-10:
            break
        
        # Generate neighbor
        move_type = rng.random()
        if move_type < 0.5:
            # Adjacent swap
            i = rng.randint(0, n_jobs - 2)
            new_perm = _swap_adjacent(current_perm, i)
        else:
            # Insert move
            i = rng.randint(0, n_jobs - 1)
            j = rng.randint(0, n_jobs - 1)
            if j == i or j == i + 1:
                continue
            job = current_perm[i]
            new_perm = _move_job(current_perm, job, j)
        
        new_ms = _makespan(new_perm, processing_times)
        delta = new_ms - current_ms
        
        if delta < 0 or rng.random() < math.exp(-delta / temp):
            current_perm = new_perm
            current_ms = new_ms
            
            if current_ms < best_ms:
                best_perm = list(current_perm)
                best_ms = current_ms
        
        temp *= cooling_rate
    
    return best_perm


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin: 10s before 120s limit
    
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    rng = random.Random(42)
    
    # Generate multiple diverse starting solutions
    candidates = []
    
    # 1. Simple LPT
    lpt_perm = list(np.argsort(-processing_times.sum(axis=1)))
    candidates.append(lpt_perm)
    
    # 2. NEH
    if time.time() < deadline:
        neh_perm = _neh_order(processing_times, rng)
        candidates.append(neh_perm)
    
    # 3. NEH with randomization (diversified)
    if time.time() < deadline:
        neh2_perm = _neh_order(processing_times, random.Random(rng.randint(0, 1000000)))
        candidates.append(neh2_perm)
    
    # 4. NEH forward-backward
    if time.time() < deadline:
        neh_fb_perm = _neh_order_forward_backward(processing_times, rng)
        candidates.append(neh_fb_perm)
    
    # Evaluate all candidates and keep the best
    best_perm = None
    best_ms = float('inf')
    for perm in candidates:
        ms = _makespan(perm, processing_times)
        if ms < best_ms:
            best_ms = ms
            best_perm = list(perm)
    
    # Local search polish on best solution
    if time.time() < deadline and n_jobs > 1:
        # Determine max iterations based on problem size and remaining time
        # First-Improvement allows for potentially more iterations if fast
        remaining = deadline - time.time()
        if n_jobs <= 30:
            max_iter = 500
        elif n_jobs <= 50:
            max_iter = 300
        elif n_jobs <= 100:
            max_iter = 100
        else:
            max_iter = 50
        
        polished = _local_search_2opt(best_perm, processing_times, max_iterations=max_iter, deadline=deadline)
        polished_ms = _makespan(polished, processing_times)
        if polished_ms < best_ms:
            best_perm = polished
            best_ms = polished_ms
    
    # Simulated annealing if time remains
    if time.time() < deadline - 5.0 and n_jobs > 1:
        sa_deadline = deadline - 5.0
        sa_perm = _simulated_annealing(best_perm, processing_times, start_time, sa_deadline, rng)
        sa_ms = _makespan(sa_perm, processing_times)
        if sa_ms < best_ms:
            best_perm = sa_perm
            best_ms = sa_ms
        
        # Final local search polish
        if time.time() < deadline - 2.0:
            final_polish = _local_search_2opt(best_perm, processing_times, max_iterations=100, deadline=deadline - 2.0)
            final_ms = _makespan(final_polish, processing_times)
            if final_ms < best_ms:
                best_perm = final_polish
                best_ms = final_ms
    
    return list(best_perm)
# EVOLVE-BLOCK-END
