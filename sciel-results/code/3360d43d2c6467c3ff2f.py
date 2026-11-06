# EVOLVE-BLOCK-START
import numpy as np
import random
import time


def _makespan(processing_times, order):
    """Compute makespan for a given job permutation."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0.0
    # completion times: C[m][j] = completion of job j on machine m
    # We process in the order of `order`
    # C for each machine is a cumulative sum over jobs in order
    # C_m[j] = max(C_{m-1}[j], C_m[j-1]) + p_{m,j}
    # More efficiently: for each machine, the completion time of job at position k
    # is max(prev machine completion of same job, current machine completion of prev job) + p
    # Let's use a 1D array for current machine completion times in job order
    # Actually, standard: C[m][pos] where pos is position in permutation
    # C[0][pos] = C[0][pos-1] + p[0][job_at_pos]
    # C[m][pos] = max(C[m-1][pos], C[m][pos-1]) + p[m][job_at_pos]
    
    # Use a 2D array but we can do it with 1D arrays for speed
    # C_prev[pos] = completion on previous machine for job at position pos
    # C_cur[pos] = completion on current machine for job at position pos
    
    C_prev = np.zeros(n_jobs)
    for m in range(n_machines):
        C_cur = np.zeros(n_jobs)
        for pos in range(n_jobs):
            job = order[pos]
            if m == 0:
                C_cur[pos] = C_cur[pos - 1] + processing_times[m, job] if pos > 0 else processing_times[m, job]
            else:
                C_cur[pos] = max(C_prev[pos], C_cur[pos - 1] if pos > 0 else 0) + processing_times[m, job]
        C_prev = C_cur
    return C_prev[-1]


def _v_shaped_init(processing_times, rng):
    """Generate initial solution using V-shaped heuristic."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Total processing time for each job
    totals = processing_times.sum(axis=1)
    
    # Sort jobs by total processing time descending
    jobs_sorted = np.argsort(-totals)
    
    # V-shape: split into two halves
    # Left half: first ceil(n/2) jobs, placed in ascending order of total (or some criterion)
    # Right half: remaining jobs, placed in descending order
    # Actually, classic V-shape: sort by total, first half ascending, second half descending
    # This creates a V when plotted
    
    half = n_jobs // 2
    left_jobs = jobs_sorted[:half]
    right_jobs = jobs_sorted[half:]
    
    # Left: place in order (already sorted descending by total, so we reverse for ascending)
    # Actually, the V-shape: high totals in middle? No.
    # Let me reconsider: In V-shaped heuristic for flow shop, jobs are divided into two sets.
    # Set 1: jobs with higher total processing time, scheduled first
    # Set 2: remaining jobs, scheduled last
    # Within each set, use some rule.
    
    # Simpler: sort all jobs by total processing time
    # Take top half, schedule them in ascending order of their first machine time
    # Take bottom half, schedule them in descending order of their last machine time
    # This creates a V-shape
    
    # Let's use: sort by total, first half in order of increasing p[0,j], second half in order of decreasing p[-1,j]
    left_sorted = sorted(left_jobs, key=lambda j: processing_times[0, j])
    right_sorted = sorted(right_jobs, key=lambda j: -processing_times[-1, j])
    
    order = list(left_sorted) + list(right_sorted)
    
    # Add some randomness for diversification
    if rng.random() < 0.3:
        # Perturb slightly
        for _ in range(n_jobs // 5):
            i, j = rng.sample(range(n_jobs), 2)
            order[i], order[j] = order[j], order[i]
    
    return order


def _tabu_search(processing_times, initial_order, time_budget, rng):
    """Tabu search with dynamic tabu list."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs <= 1:
        return initial_order
    
    order = list(initial_order)
    best_order = list(order)
    best_makespan = _makespan(processing_times, order)
    current_makespan = best_makespan
    
    # Dynamic tabu list: store (job_i, job_j) swaps or (job, from_pos, to_pos) moves
    # We'll use swap-based neighborhood for simplicity and speed
    tabu_list = {}  # key: (i, j) -> iteration when it becomes non-tabu
    tabu_tenure = 10  # initial tenure
    
    start_time = time.time()
    iteration = 0
    no_improve_count = 0
    
    while time.time() - start_time < time_budget:
        iteration += 1
        
        # Generate neighborhood: all possible swaps
        best_neighbor = None
        best_neighbor_makespan = current_makespan + 1e10  # worse than any real value
        best_move = None
        
        # To speed up, sample a subset of moves if n_jobs is large
        if n_jobs <= 50:
            candidate_pairs = [(i, j) for i in range(n_jobs) for j in range(i + 1, n_jobs)]
        else:
            # Sample pairs
            candidate_pairs = []
            num_samples = min(200, n_jobs * (n_jobs - 1) // 2)
            for _ in range(num_samples):
                i, j = rng.sample(range(n_jobs), 2)
                if i != j:
                    candidate_pairs.append((min(i, j), max(i, j)))
        
        for i, j in candidate_pairs:
            # Check tabu
            key = (i, j)
            if key in tabu_list and tabu_list[key] > iteration:
                # Tabu, but allow if it improves best (aspiration criterion)
                if current_makespan >= best_makespan:
                    continue
            else:
                # Not tabu, evaluate
                # Swap jobs at positions i and j
                order[i], order[j] = order[j], order[i]
                new_makespan = _makespan(processing_times, order)
                order[i], order[j] = order[j], order[i]  # undo
                
                if new_makespan < best_neighbor_makespan:
                    best_neighbor_makespan = new_makespan
                    best_move = (i, j)
        
        if best_move is None:
            # No valid moves found, break
            break
        
        i, j = best_move
        # Apply the move
        order[i], order[j] = order[j], order[i]
        current_makespan = best_neighbor_makespan
        
        # Update tabu list
        key = (i, j)
        # Dynamic tenure: increase when stuck, decrease when improving
        if current_makespan < best_makespan:
            tabu_tenure = max(5, tabu_tenure - 1)
            no_improve_count = 0
        else:
            tabu_tenure = min(30, tabu_tenure + 1)
            no_improve_count += 1
        
        tabu_list[key] = iteration + tabu_tenure
        
        # Clean up old tabu entries
        if iteration % 50 == 0:
            tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}
        
        # Update best
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_order = list(order)
        
        # Diversification: if stuck too long, make a random perturbation
        if no_improve_count > 30:
            # Random restart with perturbation
            for _ in range(5):
                i, j = rng.sample(range(n_jobs), 2)
                order[i], order[j] = order[j], order[i]
            current_makespan = _makespan(processing_times, order)
            no_improve_count = 0
            tabu_list = {}
            tabu_tenure = 10
    
    return best_order


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan using V-shaped init + Tabu Search."""
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    rng = random.Random(42)
    
    # Time budget: use most of the 120s, but leave buffer
    time_budget = 110.0
    
    # Generate multiple initial solutions using V-shaped heuristic
    num_inits = min(5, max(1, n_jobs // 10))
    
    best_order = None
    best_makespan = float('inf')
    
    start_time = time.time()
    
    for _ in range(num_inits):
        init_order = _v_shaped_init(processing_times, rng)
        
        # Allocate remaining time for tabu search
        elapsed = time.time() - start_time
        remaining = time_budget - elapsed
        if remaining < 5:
            break
        
        # Run tabu search with a fraction of the time
        ts_budget = remaining / (num_inits - _) * 0.8
        
        result = _tabu_search(processing_times, init_order, ts_budget, rng)
        
        ms = _makespan(processing_times, result)
        if ms < best_makespan:
            best_makespan = ms
            best_order = result
        
        # Check time
        if time.time() - start_time > time_budget * 0.7:
            break
    
    # If we have time left, do a final tabu search from the best
    elapsed = time.time() - start_time
    if best_order is not None and time_budget - elapsed > 10:
        final_budget = time_budget - elapsed - 2
        best_order = _tabu_search(processing_times, best_order, final_budget, rng)
    
    return best_order
# EVOLVE-BLOCK-END
