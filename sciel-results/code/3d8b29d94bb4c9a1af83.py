"""Improved flow-shop scheduling via NEH heuristic with fast local search and SA."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, order):
    """Compute makespan for a given job order on a flow shop."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return 0.0
    
    # Use a flat array for completion times to reduce overhead
    # c[j][m] is the completion time of job j on machine m
    # We only need the previous job's completion times and the current job's
    prev_completion = np.zeros(n_machines)
    makespan = 0.0
    
    # Localize variables for speed
    pt = processing_times
    for job_idx in order:
        job_pt = pt[job_idx]
        # Vectorized update for the current job
        # start_time[m] = max(prev_completion[m], prev_completion[m-1] + ... )? No.
        # C[j][m] = max(C[j-1][m], C[j][m-1]) + P[j][m]
        # We can compute this iteratively for the current job.
        # Since it's a loop over machines, we can't fully vectorize the dependency on m-1.
        # But we can use a simple loop.
        curr_completion = 0.0
        for m in range(n_machines):
            start_time = max(prev_completion[m], curr_completion)
            curr_completion = start_time + job_pt[m]
        prev_completion = np.maximum(prev_completion, curr_completion) # This is wrong, prev_completion is C[j-1][m], curr_completion is C[j][m]. We need to update prev_completion to be C[j][m] for the next job.
        # Actually, C[j][m] depends on C[j-1][m] and C[j][m-1].
        # So we must iterate m.
        
        # Let's rewrite the inner loop to be fast.
        curr_c = 0.0
        for m in range(n_machines):
            start_time = prev_completion[m] if prev_completion[m] > curr_c else curr_c
            curr_c = start_time + job_pt[m]
            prev_completion[m] = curr_c
        
        if curr_c > makespan:
            makespan = curr_c
            
    return makespan


def _neh_heuristic(processing_times):
    """NEH heuristic for flow shop scheduling."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    sequence = [sorted_jobs[0]]
    
    for job in sorted_jobs[1:]:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(sequence) + 1):
            candidate = sequence[:pos] + [job] + sequence[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        sequence = sequence[:best_pos] + [job] + sequence[best_pos:]
    
    return sequence


def _palmer_heuristic(processing_times):
    """Palmer heuristic: sort by sum of (n-m-1) * P[j][m]."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    weights = np.array([n_machines - m - 1 for m in range(n_machines)])
    scores = processing_times @ weights
    return list(np.argsort(-scores))


def _local_search(processing_times, initial_order, time_limit=10.0):
    """Fast first-improvement local search using insertion and swap moves."""
    n_jobs = len(initial_order)
    if n_jobs <= 1:
        return list(initial_order)
    
    current_order = list(initial_order)
    current_makespan = _makespan(processing_times, current_order)
    best_order = list(current_order)
    best_makespan = current_makespan
    
    start_time = time.time()
    
    while True:
        if time.time() - start_time > time_limit:
            break
        
        improved = False
        
        # Try insertion moves (move a single job to a different position)
        # This is often more effective than swaps for flow shop
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            
            job_to_move = current_order[i]
            
            for j in range(n_jobs):
                if j == i or j == i + 1:
                    continue
                
                # Create new order
                new_order = current_order[:i] + current_order[i+1:]
                insert_pos = j if j > i else j
                new_order = new_order[:insert_pos] + [job_to_move] + new_order[insert_pos:]
                
                new_makespan = _makespan(processing_times, new_order)
                
                if new_makespan < current_makespan - 1e-9:
                    current_order = new_order
                    current_makespan = new_makespan
                    if current_makespan < best_makespan:
                        best_makespan = current_makespan
                        best_order = list(current_order)
                    improved = True
                    break
            
            if improved:
                break
        
        if not improved:
            # Try swaps if no insertion improvement
            for i in range(n_jobs - 1):
                if time.time() - start_time > time_limit:
                    break
                for j in range(i + 1, n_jobs):
                    if time.time() - start_time > time_limit:
                        break
                    
                    new_order = current_order[:]
                    new_order[i], new_order[j] = new_order[j], new_order[i]
                    new_makespan = _makespan(processing_times, new_order)
                    
                    if new_makespan < current_makespan - 1e-9:
                        current_order = new_order
                        current_makespan = new_makespan
                        if current_makespan < best_makespan:
                            best_makespan = current_makespan
                            best_order = list(current_order)
                        improved = True
                        break
                if improved:
                    break
        
        if not improved:
            break
    
    return best_order


def _simulated_annealing(processing_times, initial_order, time_limit=30.0, rng=None):
    """Simulated annealing for flow shop scheduling."""
    n_jobs = len(initial_order)
    if n_jobs <= 1:
        return list(initial_order)
    
    if rng is None:
        rng = np.random.RandomState(42)
    
    current_order = list(initial_order)
    current_makespan = _makespan(processing_times, current_order)
    best_order = list(current_order)
    best_makespan = current_makespan
    
    start_time = time.time()
    
    # Dynamic temperature: start high, cool down
    initial_temp = max(1.0, current_makespan * 0.5)
    temp = initial_temp
    
    # Estimate number of steps we can take
    # Assume ~1000 steps per second for this problem size
    target_steps = int(time_limit * 1000)
    
    for _ in range(target_steps):
        if time.time() - start_time > time_limit:
            break
        
        # Generate a neighbor by swapping two random positions
        i, j = rng.choice(n_jobs, 2, replace=False)
        new_order = current_order[:]
        new_order[i], new_order[j] = new_order[j], new_order[i]
        new_makespan = _makespan(processing_times, new_order)
        
        delta = new_makespan - current_makespan
        
        if delta < 0 or rng.random() < np.exp(-delta / temp):
            current_order = new_order
            current_makespan = new_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_order = list(current_order)
        
        # Cool down
        temp *= 0.9999
        if temp < 1e-6:
            break
    
    return best_order


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow shop.
    Uses multiple heuristics, local search, and simulated annealing.
    """
    processing_times = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    deadline = 140.0  # 10s safety margin
    start_time = time.time()
    
    # Step 1: Generate multiple initial solutions
    initial_solutions = []
    
    # NEH
    neh_order = _neh_heuristic(processing_times)
    initial_solutions.append(neh_order)
    
    # Palmer
    palmer_order = _palmer_heuristic(processing_times)
    initial_solutions.append(palmer_order)
    
    # Random
    rng = np.random.RandomState(42)
    random_order = list(rng.permutation(n_jobs))
    initial_solutions.append(random_order)
    
    # Step 2: Local search on each initial solution
    best_order = None
    best_makespan = float('inf')
    
    # Allocate time for local search: 20% of total time
    ls_time = min(20.0, deadline * 0.2)
    
    for sol in initial_solutions:
        elapsed = time.time() - start_time
        remaining = deadline - elapsed
        if remaining < 5.0:
            break
        
        # Use a fraction of the remaining time for each LS
        ls_time_sol = min(ls_time / len(initial_solutions), remaining * 0.2)
        ls_order = _local_search(processing_times, sol, time_limit=ls_time_sol)
        ls_makespan = _makespan(processing_times, ls_order)
        
        if ls_makespan < best_makespan:
            best_makespan = ls_makespan
            best_order = list(ls_order)
    
    if best_order is None:
        return neh_order
        
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if remaining < 10.0:
        return best_order
    
    # Step 3: Simulated Annealing on the best solution
    sa_time = min(remaining * 0.8, 80.0)
    sa_order = _simulated_annealing(processing_times, best_order, time_limit=sa_time, rng=rng)
    sa_makespan = _makespan(processing_times, sa_order)
    
    if sa_makespan < best_makespan:
        best_order = list(sa_order)
        best_makespan = sa_makespan
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if remaining < 5.0:
        return best_order
    
    # Step 4: Final polish with local search
    ls_time_final = min(remaining * 0.9, 20.0)
    ls_order_final = _local_search(processing_times, best_order, time_limit=ls_time_final)
    ls_makespan_final = _makespan(processing_times, ls_order_final)
    
    if ls_makespan_final < best_makespan:
        best_order = list(ls_order_final)
        best_makespan = ls_makespan_final
    
    return best_order
# EVOLVE-BLOCK-END
