"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _calc_makespan(processing_times, perm):
    """
    Calculate makespan for a given permutation efficiently.
    Uses vectorized numpy operations where possible, but the inner loop
    over machines is sequential due to dependencies.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return 0.0
        
    completion = np.zeros(n_machines)
    
    for j in perm:
        row = processing_times[j]
        
        prev_completion = 0.0
        for m in range(n_machines):
            if m == 0:
                new_c = completion[0] + row[0]
            else:
                if completion[m] > prev_completion:
                    new_c = completion[m] + row[m]
                else:
                    new_c = prev_completion + row[m]
            
            completion[m] = new_c
            prev_completion = new_c
            
    return completion[-1]


def _calc_makespan_incremental(processing_times, perm, i, j):
    """
    Calculate the makespan change for swapping adjacent elements at positions i and i+1.
    This is a fast heuristic that only recalculates the affected part of the schedule.
    However, for flow shop, dependencies propagate forward, so we need to be careful.
    
    Actually, a full recalculation is often safer and not much slower for small N.
    But let's try a smarter approach:
    If we swap perm[i] and perm[i+1], the schedule changes from position i onwards.
    We can pre-compute the completion times up to i-1, then simulate the two orders.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    if n_jobs < 2 or i < 0 or i >= n_jobs - 1:
        return _calc_makespan(processing_times, perm)
        
    # Pre-compute completion times up to index i-1
    completion = np.zeros(n_machines)
    for k in range(i):
        job = perm[k]
        row = processing_times[job]
        prev_c = 0.0
        for m in range(n_machines):
            if m == 0:
                new_c = completion[0] + row[0]
            else:
                if completion[m] > prev_c:
                    new_c = completion[m] + row[m]
                else:
                    new_c = prev_c + row[m]
            completion[m] = new_c
            prev_c = new_c
            
    # Save state before i
    state_before = completion.copy()
    
    # Scenario 1: Original order (A, B)
    comp1 = state_before.copy()
    job_a = perm[i]
    job_b = perm[i+1]
    
    # Process job_a
    prev_c = 0.0
    for m in range(n_machines):
        if m == 0:
            new_c = comp1[0] + processing_times[job_a, 0]
        else:
            if comp1[m] > prev_c:
                new_c = comp1[m] + processing_times[job_a, m]
            else:
                new_c = prev_c + processing_times[job_a, m]
        comp1[m] = new_c
        prev_c = new_c
        
    # Process job_b
    prev_c = 0.0
    for m in range(n_machines):
        if m == 0:
            new_c = comp1[0] + processing_times[job_b, 0]
        else:
            if comp1[m] > prev_c:
                new_c = comp1[m] + processing_times[job_b, m]
            else:
                new_c = prev_c + processing_times[job_b, m]
        comp1[m] = new_c
        prev_c = new_c
        
    # Continue with rest of jobs
    for k in range(i+2, n_jobs):
        job = perm[k]
        row = processing_times[job]
        prev_c = 0.0
        for m in range(n_machines):
            if m == 0:
                new_c = comp1[0] + row[0]
            else:
                if comp1[m] > prev_c:
                    new_c = comp1[m] + row[m]
                else:
                    new_c = prev_c + row[m]
            comp1[m] = new_c
            prev_c = new_c
            
    makespan1 = comp1[-1]
    
    # Scenario 2: Swapped order (B, A)
    comp2 = state_before.copy()
    
    # Process job_b
    prev_c = 0.0
    for m in range(n_machines):
        if m == 0:
            new_c = comp2[0] + processing_times[job_b, 0]
        else:
            if comp2[m] > prev_c:
                new_c = comp2[m] + processing_times[job_b, m]
            else:
                new_c = prev_c + processing_times[job_b, m]
        comp2[m] = new_c
        prev_c = new_c
        
    # Process job_a
    prev_c = 0.0
    for m in range(n_machines):
        if m == 0:
            new_c = comp2[0] + processing_times[job_a, 0]
        else:
            if comp2[m] > prev_c:
                new_c = comp2[m] + processing_times[job_a, m]
            else:
                new_c = prev_c + processing_times[job_a, m]
        comp2[m] = new_c
        prev_c = new_c
        
    # Continue with rest of jobs
    for k in range(i+2, n_jobs):
        job = perm[k]
        row = processing_times[job]
        prev_c = 0.0
        for m in range(n_machines):
            if m == 0:
                new_c = comp2[0] + row[0]
            else:
                if comp2[m] > prev_c:
                    new_c = comp2[m] + row[m]
                else:
                    new_c = prev_c + row[m]
            comp2[m] = new_c
            prev_c = new_c
            
    makespan2 = comp2[-1]
    
    return min(makespan1, makespan2)


def _swap_adjacent(perm, i):
    """Swap adjacent elements at positions i and i+1."""
    new_perm = perm.copy()
    new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
    return new_perm


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    # 1. Initial Construction: NEH Heuristic
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    current_perm = [sorted_jobs[0]]
    
    for i in range(1, n_jobs):
        job = sorted_jobs[i]
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_perm) + 1):
            candidate_perm = current_perm[:pos] + [job] + current_perm[pos:]
            ms = _calc_makespan(processing_times, candidate_perm)
            if ms < best_makespan:
                best_makespan = ms
                best_pos =pos
                
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
        
    best_perm = current_perm
    best_ms = _calc_makespan(processing_times, best_perm)
    
    # 2. Local Search: Adjacent Swaps
    # Use adjacent swaps to refine the schedule.
    # This is a 1-move neighborhood which is smaller but allows for more iterations.
    # We use a first-improvement strategy to move quickly.
    
    improved = True
    while improved and (time.time() < deadline - 2.0):
        improved = False
        
        # Try all adjacent swaps
        for i in range(n_jobs - 1):
            if time.time() > deadline - 5.0:
                return best_perm
                
            # Check if swapping i and i+1 improves makespan
            # We can use the incremental evaluation or just full recalculation
            # Full recalculation is simpler and robust
            new_perm = _swap_adjacent(best_perm, i)
            ms = _calc_makespan(processing_times, new_perm)
            
            if ms < best_ms - 1e-9:
                best_ms = ms
                best_perm = new_perm
                improved = True
                # Restart the scan from the beginning to ensure we find all improvements
                # Or we could continue, but restarting is safer for convergence
                break
                
    return best_perm
# EVOLVE-BLOCK-END
