"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, processing_times):
    """
    Compute the makespan for a given permutation of job indices.
    Uses the standard DP: C[j,m] = max(C[j,m-1], C[j-1,m]) + P[j,m]
    Returns the completion time of the last job on the last machine.
    """
    n_machines = processing_times.shape[1]
    if len(perm) == 0:
        return 0.0
    
    # prev_job_completion[m] = completion time of previous job on machine m
    prev_job_completion = [0.0] * n_machines
    
    for job in perm:
        # current_job_completion_on_prev = completion time of current job on machine m-1
        current_job_completion_on_prev = 0.0
        for m in range(n_machines):
            p = processing_times[job, m]
            if m == 0:
                start = prev_job_completion[0]
            else:
                start = current_job_completion_on_prev if current_job_completion_on_prev > prev_job_completion[m] else prev_job_completion[m]
            current_job_completion_on_prev = start + p
            prev_job_completion[m] = current_job_completion_on_prev
    
    return prev_job_completion[-1]


def _makespan_swap(perm, processing_times, i, j):
    """
    Compute makespan after swapping positions i and j in perm.
    Optimized: only recompute affected portion.
    For adjacent swap, this is O(n_machines * 2) instead of O(n_jobs * n_machines).
    """
    n_machines = processing_times.shape[1]
    n_jobs = len(perm)
    
    if i == j:
        return _makespan(perm, processing_times)
    
    if abs(i - j) > 1:
        # Non-adjacent: use full recomputation
        new_perm = list(perm)
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        return _makespan(new_perm, processing_times)
    
    # Adjacent swap: recompute from position min(i,j) to end
    # First compute completion times up to position min(i,j)
    start_pos = min(i, j)
    
    # Compute completion times for jobs 0..start_pos-1
    prev_job_completion = [0.0] * n_machines
    for pos in range(start_pos):
        job = perm[pos]
        current_job_completion_on_prev = 0.0
        for m in range(n_machines):
            p = processing_times[job, m]
            if m == 0:
                start = prev_job_completion[0]
            else:
                start = current_job_completion_on_prev if current_job_completion_on_prev > prev_job_completion[m] else prev_job_completion[m]
            current_job_completion_on_prev = start + p
            prev_job_completion[m] = current_job_completion_on_prev
    
    # Now process the swapped jobs
    # Original: perm[i], perm[j]
    # Swapped: perm[j], perm[i]
    for pos in range(start_pos, n_jobs):
        if pos == i:
            job = perm[j]
        elif pos == j:
            job = perm[i]
        else:
            job = perm[pos]
        
        current_job_completion_on_prev = 0.0
        for m in range(n_machines):
            p = processing_times[job, m]
            if m == 0:
                start = prev_job_completion[0]
            else:
                start = current_job_completion_on_prev if current_job_completion_on_prev > prev_job_completion[m] else prev_job_completion[m]
            current_job_completion_on_prev = start + p
            prev_job_completion[m] = current_job_completion_on_prev
    
    return prev_job_completion[-1]


def _makespan_move(perm, processing_times, i, j):
    """
    Compute makespan after moving job at position i to position j.
    If i < j: remove from i, insert at j (shifts i+1..j left by 1)
    If i > j: remove from i, insert at j (shifts j..i-1 right by 1)
    """
    n_machines = processing_times.shape[1]
    n_jobs = len(perm)
    
    if i == j:
        return _makespan(perm, processing_times)
    
    # Build the new permutation
    job_to_move = perm[i]
    
    # Create new perm without job at i
    new_perm = []
    for pos in range(n_jobs):
        if pos == i:
            continue
        new_perm.append(perm[pos])
    
    # Insert job_to_move at position j
    # If i < j, the effective insertion position in new_perm is j-1
    # If i > j, the effective insertion position in new_perm is j
    if i < j:
        insert_pos = j - 1
    else:
        insert_pos = j
    
    new_perm.insert(insert_pos, job_to_move)
    
    return _makespan(new_perm, processing_times)


def _neh_initialization(processing_times):
    """
    NEH heuristic for initial permutation.
    1. Sort jobs by total processing time in descending order.
    2. Insert each job into the position that minimizes the makespan.
    """
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    # Step 1: Sort by total processing time
    total_processing = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-total_processing))
    
    # Step 2: Insert one by one
    current_perm = []
    
    for job in sorted_jobs:
        best_pos = 0
        best_makespan = float('inf')
        
        # Try inserting at each position
        for pos in range(len(current_perm) + 1):
            new_perm = current_perm.copy()
            new_perm.insert(pos, job)
            ms = _makespan(new_perm, processing_times)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_perm.insert(best_pos, job)
        
    return current_perm


def _tabu_search(initial_perm, processing_times, time_limit=10.0):
    """
    Tabu search with adjacent swaps and moves.
    Uses aspiration criteria and dynamic tabu tenure.
    """
    n_jobs = len(initial_perm)
    if n_jobs <= 1:
        return initial_perm
    
    current_perm = list(initial_perm)
    current_makespan = _makespan(current_perm, processing_times)
    best_perm = list(current_perm)
    best_makespan = current_makespan
    
    # Tabu list: dictionary mapping move key to expiration iteration
    tabu_list = {}
    base_tabu_tenure = max(10, n_jobs // 2)
    
    start_time = time.time()
    iteration = 0
    no_improve_count = 0
    max_no_improve = 80
    
    # Frequency memory for diversification
    move_freq = {}
    
    while time.time() - start_time < time_limit:
        iteration += 1
        best_neighbor_perm = None
        best_neighbor_makespan = current_makespan
        best_neighbor_move = None
        
        # Generate neighborhood
        moves_to_try = []
        
        # Always include all adjacent swaps
        for i in range(n_jobs - 1):
            moves_to_try.append(('swap', i, i + 1))
        
        # Add random moves based on problem size
        n_random_moves = min(200, max(50, n_jobs * 5))
        for _ in range(n_random_moves):
            i = np.random.randint(0, n_jobs)
            j = np.random.randint(0, n_jobs)
            if i != j:
                moves_to_try.append(('move', i, j))
        
        for move in moves_to_try:
            move_type = move[0]
            i, j = move[1], move[2]
            
            if move_type == 'swap':
                if i > j:
                    i, j = j, i
                move_key = ('swap', i, j)
            else:
                if i == j:
                    continue
                move_key = ('move', i, j)
            
            # Compute new makespan
            if move_type == 'swap':
                new_makespan = _makespan_swap(current_perm, processing_times, i, j)
            else:
                new_makespan = _makespan_move(current_perm, processing_times, i, j)
            
            # Check tabu status
            is_tabu = move_key in tabu_list and tabu_list[move_key] > iteration
            is_aspiration = new_makespan < best_makespan
            
            if is_tabu and not is_aspiration:
                continue
            
            # Apply diversification: penalize frequently used moves
            freq_penalty = move_freq.get(move_key, 0) * 0.01
            adjusted_makespan = new_makespan + freq_penalty
            
            if adjusted_makespan < best_neighbor_makespan:
                best_neighbor_makespan = adjusted_makespan
                best_neighbor_perm = None  # We'll reconstruct
                best_neighbor_move = move_key
                best_neighbor_actual_makespan = new_makespan
        
        if best_neighbor_move is None:
            # No improving move found
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
            
            # Perturb: apply a random sequence of moves
            n_perturb = max(3, n_jobs // 5)
            for _ in range(n_perturb):
                i = np.random.randint(0, n_jobs)
                j = np.random.randint(0, n_jobs)
                if i != j:
                    job_to_move = current_perm.pop(i)
                    if i < j:
                        j -= 1
                    current_perm.insert(j, job_to_move)
            
            current_makespan = _makespan(current_perm, processing_times)
        else:
            no_improve_count = 0
            
            # Update tabu list
            tabu_tenure = base_tabu_tenure + np.random.randint(-3, 4)
            tabu_list[best_neighbor_move] = iteration + tabu_tenure
            
            # Update frequency memory
            move_freq[best_neighbor_move] = move_freq.get(best_neighbor_move, 0) + 1
            
            # Clean up tabu list periodically
            if iteration % 50 == 0:
                for key in list(tabu_list.keys()):
                    if tabu_list[key] <= iteration:
                        del tabu_list[key]
            
            # Apply the best move
            move_type, i, j = best_neighbor_move
            if move_type == 'swap':
                current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
            else:
                job_to_move = current_perm.pop(i)
                if i < j:
                    j -= 1
                current_perm.insert(j, job_to_move)
            
            current_makespan = best_neighbor_actual_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_perm = list(current_perm)
    
    return best_perm


def _multiple_restart_tabu(processing_times, initial_perm, time_limit=25.0):
    """
    Run tabu search with multiple restarts.
    """
    n_jobs = len(processing_times)
    
    # Determine number of restarts based on time
    if n_jobs <= 20:
        n_restarts = 5
        per_restart_time = time_limit / n_restarts
    elif n_jobs <= 50:
        n_restarts = 3
        per_restart_time = time_limit / n_restarts
    else:
        n_restarts = 2
        per_restart_time = time_limit / n_restarts
    
    best_perm = list(initial_perm)
    best_makespan = _makespan(best_perm, processing_times)
    
    start_time = time.time()
    
    for restart in range(n_restarts):
        if restart == 0:
            current_initial = list(initial_perm)
        else:
            # Perturb the best solution for restart
            current_initial = list(best_perm)
            n_perturb = max(5, n_jobs // 3)
            for _ in range(n_perturb):
                i = np.random.randint(0, n_jobs)
                j = np.random.randint(0, n_jobs)
                if i != j:
                    job_to_move = current_initial.pop(i)
                    if i < j:
                        j -= 1
                    current_initial.insert(j, job_to_move)
        
        remaining_time = time_limit - (time.time() - start_time)
        if remaining_time < 2.0:
            break
        
        perm = _tabu_search(current_initial, processing_times, time_limit=min(per_restart_time, remaining_time))
        ms = _makespan(perm, processing_times)
        
        if ms < best_makespan:
            best_makespan = ms
            best_perm = list(perm)
    
    return best_perm


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on local synthetic flow-shop instances.
    Uses NEH initialization and multi-restart tabu search with aspiration.
    """
    if len(processing_times) == 0:
        return []
    
    n_jobs = len(processing_times)
    
    # Initial solution: NEH
    initial_perm = _neh_initialization(processing_times)
    
    # Adjust time limit based on problem size
    if n_jobs <= 20:
        time_limit = 25.0
    elif n_jobs <= 50:
        time_limit = 20.0
    else:
        time_limit = 15.0
    
    # Run multi-restart tabu search
    final_perm = _multiple_restart_tabu(processing_times, initial_perm, time_limit=time_limit)
    
    return list(final_perm)
# EVOLVE-BLOCK-END
