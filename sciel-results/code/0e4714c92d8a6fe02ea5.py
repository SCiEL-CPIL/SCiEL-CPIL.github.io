"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _calc_makespan(processing_times, perm):
    """
    Calculate makespan for a given permutation.
    Optimized for small N and M using a tight Python loop.
    """
    n_jobs = len(perm)
    if n_jobs == 0:
        return 0.0
    
    n_machines = processing_times.shape[1]
    
    if n_machines == 1:
        return np.sum(processing_times[perm, 0])
        
    completion = [0.0] * n_machines
    
    for job_idx in perm:
        row = processing_times[job_idx]
        prev_c = 0.0
        for m in range(n_machines):
            if completion[m] > prev_c:
                new_c = completion[m] + row[m]
            else:
                new_c = prev_c + row[m]
            completion[m] = new_c
            prev_c = new_c
            
    return completion[-1]

def _neh_heuristic(processing_times):
    """
    NEH (Nawaz-Enscore-Ham) heuristic for initial solution construction.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Sort jobs by total processing time descending
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    current_perm = [sorted_jobs[0]]
    
    # Insert each subsequent job at the best position
    for i in range(1, n_jobs):
        job = sorted_jobs[i]
        best_pos = 0
        best_ms = float('inf')
        
        for pos in range(len(current_perm) + 1):
            # Create candidate permutation
            candidate = current_perm[:pos] + [job] + current_perm[pos:]
            ms = _calc_makespan(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
                
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
        
    return current_perm

def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # 1. Initial Construction: NEH Heuristic
    current_perm = _neh_heuristic(processing_times)
    best_perm = current_perm[:]
    best_ms = _calc_makespan(processing_times, best_perm)
    
    # 2. Tabu Search
    # Tabu list stores the reverse move (i, j) to prevent immediate oscillation
    tabu_list = {} # {(i,j): expiration_iteration}
    tabu_tenure = max(5, int(n_jobs * 0.5))
    
    iteration = 0
    no_improve = 0
    max_no_improve = 500
    
    current_ms = best_ms
    
    while time.time() < deadline - 1.0:
        iteration += 1
        
        # Generate neighbor: swap two jobs
        i = random.randint(0, n_jobs - 1)
        j = random.randint(0, n_jobs - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
            
        move_key = (i, j)
        
        # Check Tabu
        is_tabu = False
        if move_key in tabu_list:
            if tabu_list[move_key] > iteration:
                is_tabu = True
        
        # Perform swap
        new_perm = current_perm[:]
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        new_ms = _calc_makespan(processing_times, new_perm)
        
        # Aspiration criterion: accept if it improves the best known solution
        # OR if it's not tabu
        if new_ms < best_ms or not is_tabu:
            # Accept move
            current_perm = new_perm
            current_ms = new_ms
            
            # Add to tabu list
            tabu_list[move_key] = iteration + tabu_tenure
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm[:]
                no_improve = 0
            else:
                no_improve += 1
                
            # Diversification: if stuck for a long time, increase tenure or clear tabu?
            # Let's just increase tenure slightly if stuck
            if no_improve > max_no_improve:
                tabu_tenure = int(tabu_tenure * 1.2)
                no_improve = 0
                # Optional: random restart? 
                # Let's try a random restart from best_perm if stuck too long
                if tabu_tenure > n_jobs * 2:
                    # Restart
                    current_perm = best_perm[:]
                    current_ms = best_ms
                    tabu_list = {}
                    tabu_tenure = max(5, int(n_jobs * 0.5))
        
        else:
            # Move is tabu and not improving best, so reject
            pass
            
    return best_perm
# EVOLVE-BLOCK-END
