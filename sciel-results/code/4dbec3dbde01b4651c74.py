# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(perm, processing_times):
    """
    Optimized makespan calculation for flow shop.
    """
    n_jobs, n_machines = processing_times.shape
    # completion[m] is the time when machine m becomes free
    completion = [0.0] * n_machines
    
    for job in perm:
        p = processing_times[job]
        prev_finish = 0.0
        for m in range(n_machines):
            # Start time is max(machine free time, job finished previous machine)
            start = completion[m] if completion[m] > prev_finish else prev_finish
            finish = start + p[m]
            completion[m] = finish
            prev_finish = finish
            
    return completion[-1]

def _neh_construct(processing_times, noise=0.0):
    """
    Construct a permutation using the NEH heuristic.
    If noise > 0, adds random noise to total processing times to diversify.
    """
    n_jobs, n_machines = processing_times.shape
    totals = np.sum(processing_times, axis=1)
    
    if noise > 0:
        noise_vec = np.random.uniform(-noise, noise, n_jobs)
        totals = totals + noise_vec
        
    # Sort by total processing time descending
    sorted_indices = np.argsort(-totals)
    
    seq = [sorted_indices[0]]
    
    for idx in sorted_indices[1:]:
        best_pos = 0
        best_cost = float('inf')
        best_seq = seq[:]
        
        for pos in range(len(seq) + 1):
            # Create new sequence
            new_seq = seq[:pos] + [idx] + seq[pos:]
            cost = _makespan(new_seq, processing_times)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
                best_seq = new_seq
                
        seq = best_seq
        
    return seq

def _local_search_oropt(perm, processing_times, time_limit, start_time):
    """
    Local search using Or-opt moves (move a block of 1, 2, or 3 jobs).
    """
    n_jobs = len(perm)
    current_perm = perm[:]
    current_cost = _makespan(current_perm, processing_times)
    
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try block sizes 1, 2, 3
        for k in range(1, 4):
            if k > n_jobs:
                continue
            for i in range(n_jobs - k + 1):
                if (time.time() - start_time) > time_limit:
                    return current_perm, current_cost
                
                block = current_perm[i:i+k]
                # Create remaining list
                remaining = current_perm[:i] + current_perm[i+k:]
                
                # Try all insertion positions
                for j in range(len(remaining) + 1):
                    # Skip original position and adjacent to avoid trivial moves
                    # Original position in remaining would be i if j < i, or i+k-1?
                    # It's easier to just check if the resulting perm is the same
                    # But checking makespan is cheap enough if we just try all j
                    
                    new_perm = remaining[:j] + block + remaining[j:]
                    
                    # Optional: Skip if new_perm == current_perm
                    # This check is O(N), makespan is O(N*M). 
                    # For small N, it's fine.
                    
                    new_cost = _makespan(new_perm, processing_times)
                    
                    if new_cost < current_cost:
                        current_perm = new_perm
                        current_cost = new_cost
                        improved = True
                        # Restart local search from this new point
                        # Break out of loops
                        break
                if improved:
                    break
            if improved:
                break
                
    return current_perm, current_cost

def order_jobs(processing_times):
    n_jobs, n_machines = processing_times.shape
    
    # Time management
    start_time = time.time()
    time_limit = 115.0  # Leave 5s buffer
    
    # 1. Initial Construction with NEH
    best_perm = _neh_construct(processing_times)
    best_cost = _makespan(best_perm, processing_times)
    
    current_perm = best_perm[:]
    current_cost = best_cost
    
    # 2. Iterated Local Search
    # We will perturb the current solution and run local search again.
    # If the new solution is better, we accept it.
    
    while (time.time() - start_time) < time_limit:
        # Run Local Search on current_perm
        current_perm, current_cost = _local_search_oropt(current_perm, processing_times, time_limit, start_time)
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_perm = current_perm[:]
            
        # Perturbation
        # Randomly swap a few jobs to move away from local optimum
        perturbed_perm = current_perm[:]
        # Number of swaps depends on problem size
        num_swaps = random.randint(2, max(2, n_jobs // 10))
        
        for _ in range(num_swaps):
            i = random.randint(0, n_jobs - 1)
            j = random.randint(0, n_jobs - 1)
            if i != j:
                perturbed_perm[i], perturbed_perm[j] = perturbed_perm[j], perturbed_perm[i]
                
        current_perm = perturbed_perm
        # Recalculate cost for the perturbed solution (optional, LS will do it, but good for tracking)
        # current_cost = _makespan(current_perm, processing_times)
        
        # Check time again before next LS
        if (time.time() - start_time) > time_limit:
            break
            
    return best_perm
# EVOLVE-BLOCK-END
