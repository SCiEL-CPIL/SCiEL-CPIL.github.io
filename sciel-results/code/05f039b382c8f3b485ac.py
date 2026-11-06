"""Iterated Local Search with Or-opt and Insertion moves for permutation flow-shop."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(perm, pt):
    """Compute makespan for a given permutation."""
    n_jobs = len(perm)
    if n_jobs == 0:
        return 0.0
    n_machines = pt.shape[1]
    
    # C[j, m] = max(C[j-1, m], C[j, m-1]) + pt[perm[j], m]
    # We only need the completion time of the previous job on each machine.
    prev_row = np.zeros(n_machines)
    
    for j in range(n_jobs):
        job = perm[j]
        current_cum = 0.0
        for m in range(n_machines):
            current_cum = max(prev_row[m], current_cum) + pt[j, m]
            prev_row[m] = current_cum
            
    return prev_row[-1]

def _generate_neh_order(pt):
    """Generate an initial ordering using the NEH heuristic."""
    n_jobs, n_machines = pt.shape
    
    # Sum processing times for each job
    totals = pt.sum(axis=1)
    
    # Sort jobs in descending order of total processing time
    sorted_jobs = list(np.argsort(-totals))
    
    # Build the permutation incrementally using NEH
    perm = []
    
    for job in sorted_jobs:
        if not perm:
            perm.append(job)
            continue
        
        # Find the best position to insert this job
        best_pos = 0
        best_ms = float('inf')
        
        # Try all possible insertion positions
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(candidate, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        perm.insert(best_pos, job)
    
    return perm

def _local_search(perm, pt, time_limit, deadline):
    """
    Perform local search using Or-opt (blocks of size 1, 2, 3) and Insertion moves.
    Returns the best permutation found.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm

    current_perm = list(perm)
    current_ms = _makespan(current_perm, pt)
    
    best_perm = list(current_perm)
    best_ms = current_ms

    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 2000  # Stop local search if no improvement for a while

    # Block sizes for Or-opt
    block_sizes = [1, 2, 3]
    
    while time.time() - start_time < time_limit:
        if time.time() > deadline:
            break
            
        improved = False
        
        # Try Or-opt moves
        for size in block_sizes:
            if n_jobs <= size:
                continue
                
            # Randomly select a starting position for the block
            # To ensure coverage, we can iterate or sample. Sampling is faster for large n.
            # For small n, iterating is better. Let's mix: if n < 50, iterate, else sample.
            if n_jobs < 50:
                starts = range(n_jobs - size)
            else:
                # Sample a few starts
                starts = np.random.choice(n_jobs - size, size=min(20, n_jobs - size), replace=False)
            
            for i in starts:
                if time.time() > deadline:
                    break
                
                block = current_perm[i:i+size]
                
                # Remove block
                remaining = current_perm[:i] + current_perm[i+size:]
                
                # Try all insertion positions in the remaining list
                # Optimization: Only check positions that are not the original one (i)
                # and perhaps limit to a window if n is large? 
                # For now, check all positions in remaining.
                best_pos_in_remaining = -1
                best_ms_candidate = current_ms
                
                # We need to evaluate insertion at pos p in 'remaining'
                # This creates a new perm of length n_jobs
                # We can optimize by checking only a subset of positions if n is large, 
                # but let's start with all for correctness, then optimize if slow.
                
                # Actually, for speed, we can limit the search space.
                # But let's try full first. If n=100, 100 evals per move. 
                # 100 moves * 100 evals * 1000 steps = 10^7 evals. Might be too slow in Python.
                
                # Let's restrict insertion positions to a window around the original position
                # or just random positions?
                # Standard ILS often uses best improvement or first improvement.
                # Let's use a restricted neighborhood for speed: 
                # Insert near the original position or at random positions.
                
                # Let's try inserting at all positions but break early if we find a good one?
                # No, best improvement is better for local search quality.
                
                # To speed up, we can sample insertion positions if n is large.
                if n_jobs > 100:
                    # Sample 10 insertion positions
                    pos_candidates = np.random.choice(len(remaining), size=min(10, len(remaining)), replace=False)
                else:
                    pos_candidates = range(len(remaining))
                    
                for p in pos_candidates:
                    candidate = remaining[:p] + block + remaining[p:]
                    ms = _makespan(candidate, pt)
                    if ms < best_ms_candidate:
                        best_ms_candidate = ms
                        best_pos_in_remaining = p
                        
                if best_pos_in_remaining != -1:
                    # Apply the move
                    remaining = current_perm[:i] + current_perm[i+size:]
                    current_perm = remaining[:best_pos_in_remaining] + block + remaining[best_pos_in_remaining:]
                    current_ms = best_ms_candidate
                    
                    if current_ms < best_ms:
                        best_ms = current_ms
                        best_perm = list(current_perm)
                        improved = True
                    else:
                        # It's an improvement over previous current, but not best.
                        # Still accept it as current state for local search?
                        # Yes, standard local search accepts any improvement.
                        improved = True
                        # Break out of block loop to restart with new permutation
                        break 
                        
            if improved:
                break # Restart block size loop with new permutation
                
        if not improved:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
        else:
            no_improve_count = 0
            
    return best_perm

def _perturb(perm, strength):
    """
    Perturb the permutation by performing a series of random Or-opt moves.
    strength: number of moves.
    """
    n_jobs = len(perm)
    perm = list(perm)
    
    for _ in range(strength):
        if n_jobs <= 1:
            break
        size = np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
        if n_jobs <= size:
            size = 1
            
        i = np.random.randint(0, n_jobs - size)
        block = perm[i:i+size]
        remaining = perm[:i] + perm[i+size:]
        
        # Insert at a random position
        p = np.random.randint(0, len(remaining) + 1)
        perm = remaining[:p] + block + remaining[p:]
        
    return perm

def _iterated_local_search(initial_perm, pt, time_limit, deadline):
    """
    ILS: Local Search -> Perturb -> Local Search -> ...
    """
    current_perm = list(initial_perm)
    current_ms = _makespan(current_perm, pt)
    
    best_perm = list(current_perm)
    best_ms = current_ms
    
    start_time = time.time()
    
    # Adaptive perturbation strength
    # Start small, increase if stuck
    strength = 2
    
    while time.time() < time_limit:
        if time.time() > deadline:
            break
            
        # 1. Local Search
        # Allocate a small time slice to local search, e.g., 0.1s or until convergence
        ls_time_limit = min(0.5, time_limit - (time.time() - start_time))
        if ls_time_limit <= 0.1:
            break
            
        ls_perm = _local_search(current_perm, pt, ls_time_limit, deadline)
        ls_ms = _makespan(ls_perm, pt)
        
        if ls_ms < best_ms:
            best_ms = ls_ms
            best_perm = list(ls_perm)
            current_perm = ls_perm
            current_ms = ls_ms
            # Reset strength if we improved
            strength = 2
        else:
            # No improvement from local search on this perturbed solution
            # Increase perturbation strength
            strength += 1
            if strength > 10:
                strength = 10
            
            # Perturb the best solution found so far (or current?)
            # Standard ILS: Perturb the current local optimum.
            # If current == best, perturb best.
            current_perm = _perturb(best_perm, strength)
            current_ms = _makespan(current_perm, pt)
            
    return best_perm

def order_jobs(processing_times):
    """Main entry point: order jobs to minimize makespan on a flow shop."""
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Deadline guard
    start_total = time.time()
    deadline = 145.0 # Safety margin from 150s kill
    
    # 1. Generate initial solution using NEH
    neh_perm = _generate_neh_order(processing_times)
    neh_ms = _makespan(neh_perm, processing_times)
    
    best_perm = neh_perm
    best_ms = neh_ms
    
    elapsed = time.time() - start_total
    time_left = deadline - elapsed
    
    if time_left > 1.0:
        # 2. Run Iterated Local Search
        ils_perm = _iterated_local_search(neh_perm, processing_times, time_left - 5.0, deadline)
        ils_ms = _makespan(ils_perm, processing_times)
        
        if ils_ms < best_ms:
            best_perm = ils_perm
            best_ms = ils_ms
            
    return best_perm
# EVOLVE-BLOCK-END
