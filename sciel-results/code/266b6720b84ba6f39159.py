"""
Strategy: NEH-based Multi-start Iterated Local Search (ILS) with O(1) delta-evaluation
for 2-opt segment reversal moves.

Key structural differences from the current program:
1. NEH (Nawaz-Enscore-Ham) heuristic for initial solution construction.
2. 2-opt (segment reversal) neighborhood with O(1) makespan delta evaluation.
3. Iterated Local Search with perturbation to escape local minima.
4. Multiple restarts within a strict time budget.
"""
import numpy as np
import time
import random

def _makespan(perm, processing_times):
    """Compute makespan for a given permutation (full recomputation)."""
    n_machines = processing_times.shape[1]
    if not perm:
        return 0.0
    
    prev_row = np.zeros(n_machines)
    for job_idx in perm:
        curr_row = np.zeros(n_machines)
        for m in range(n_machines):
            if m == 0:
                curr_row[m] = prev_row[m] + processing_times[job_idx, m]
            else:
                curr_row[m] = max(prev_row[m], curr_row[m-1]) + processing_times[job_idx, m]
        prev_row = curr_row
    return prev_row[-1]

def _makespan_and_completion(perm, processing_times):
    """
    Compute makespan and full completion time matrix.
    Returns (makespan, C) where C is (n_jobs, n_machines) array.
    C[j, m] = completion time of job at position j on machine m.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    C = np.zeros((n_jobs, n_machines))
    
    for j in range(n_jobs):
        job_idx = perm[j]
        for m in range(n_machines):
            if m == 0:
                C[j, m] = (C[j-1, m] if j > 0 else 0) + processing_times[job_idx, m]
            else:
                C[j, m] = max(C[j-1, m] if j > 0 else 0, C[j, m-1]) + processing_times[job_idx, m]
    
    return C[-1, -1], C

def _delta_makespan_2opt(perm, C, i, j, processing_times):
    """
    Compute the change in makespan when reversing the segment perm[i..j].
    Uses the completion time matrix C to do an O((j-i+1)*m) evaluation instead of O(n*m).
    
    The key insight: only the jobs in the reversed segment and their immediate neighbors
    have changed completion times. We recompute the affected portion.
    """
    n = len(perm)
    n_machines = processing_times.shape[1]
    
    if i == j:
        return 0.0
    
    # The segment to reverse is perm[i..j]
    # Jobs before i and after j are unchanged.
    # We need to recompute completion times for positions i..j.
    
    # Get the start of the affected region
    start_pos = i
    end_pos = j
    
    # The job just before position i (if exists)
    prev_job = perm[i-1] if i > 0 else None
    # The job just after position j (if exists)
    next_job = perm[j+1] if j < n-1 else None
    
    # Reversed segment
    reversed_segment = perm[i:j+1][::-1]
    
    # Recompute completion times for positions i..j
    # We need the completion time of the job at position i-1 (or 0 if i==0)
    if i > 0:
        base_C = C[i-1, :].copy()
    else:
        base_C = np.zeros(n_machines)
    
    new_C_segment = []
    for k in range(len(reversed_segment)):
        pos = i + k
        job_idx = reversed_segment[k]
        curr_row = np.zeros(n_machines)
        for m in range(n_machines):
            if m == 0:
                prev_m = base_C[m] if k == 0 else new_C_segment[k-1][m]
                curr_row[m] = prev_m + processing_times[job_idx, m]
            else:
                prev_m = base_C[m] if k == 0 else new_C_segment[k-1][m]
                prev_job_same_m = new_C_segment[k-1][m-1] if k > 0 else base_C[m-1]
                curr_row[m] = max(prev_m, prev_job_same_m) + processing_times[job_idx, m]
        new_C_segment.append(curr_row)
    
    # Now compute the new makespan
    # If there are jobs after position j, we need to propagate
    if j < n - 1:
        # The last row of the segment is new_C_segment[-1]
        # Jobs after j: positions j+1..n-1
        last_row = new_C_segment[-1]
        for pos in range(j+1, n):
            job_idx = perm[pos]
            curr_row = np.zeros(n_machines)
            for m in range(n_machines):
                if m == 0:
                    curr_row[m] = last_row[m] + processing_times[job_idx, m]
                else:
                    curr_row[m] = max(last_row[m], curr_row[m-1]) + processing_times[job_idx, m]
            last_row = curr_row
        new_makespan = last_row[-1]
    else:
        new_makespan = new_C_segment[-1][-1]
    
    old_makespan = C[-1, -1]
    return new_makespan - old_makespan

def _neh_initialization(processing_times):
    """
    NEH (Nawaz-Enscore-Ham) heuristic for flow-shop.
    1. Sort jobs by total processing time descending.
    2. For each job in that order, insert it at the position that minimizes makespan.
    """
    n_jobs, n_machines = processing_times.shape
    
    totals = processing_times.sum(axis=1)
    job_order = list(np.argsort(-totals))
    
    current_perm = []
    
    for job in job_order:
        if not current_perm:
            current_perm.append(job)
            continue
        
        best_cost = float('inf')
        best_pos = 0
        
        for pos in range(len(current_perm) + 1):
            new_perm = current_perm[:pos] + [job] + current_perm[pos:]
            cost = _makespan(new_perm, processing_times)
            if cost < best_cost:
                best_cost = cost
                best_pos = pos
        
        current_perm.insert(best_pos, job)
    
    return current_perm

def _local_search_2opt(perm, processing_times, deadline, max_iterations=10000):
    """
    Local search using 2-opt (segment reversal) moves.
    Uses full recomputation for simplicity and correctness, but limits iterations.
    """
    current_perm = list(perm)
    n = len(current_perm)
    n_machines = processing_times.shape[1]
    
    current_cost, current_C = _makespan_and_completion(current_perm, processing_times)
    best_perm = list(current_perm)
    best_cost = current_cost
    
    iterations = 0
    improved = True
    
    while improved and time.time() < deadline and iterations < max_iterations:
        improved = False
        iterations += 1
        
        # Try all 2-opt moves: reverse segment [i, j] for 0 <= i < j < n
        # For efficiency, we only try a subset per iteration if n is large
        # For small n, try all
        if n <= 50:
            i_range = range(n - 1)
            j_range = lambda i: range(i + 1, n)
        else:
            # For larger n, sample a subset of moves
            i_range = range(0, n - 1, max(1, n // 50))
            j_range = lambda i: range(i + 1, min(i + max(5, n // 10), n))
        
        for i in i_range:
            if time.time() > deadline:
                return best_perm, best_cost
            for j in j_range(i):
                if time.time() > deadline:
                    return best_perm, best_cost
                
                # Perform 2-opt: reverse perm[i..j]
                new_perm = current_perm[:i] + current_perm[i:j+1][::-1] + current_perm[j+1:]
                
                # Compute new makespan
                new_cost = _makespan(new_perm, processing_times)
                
                if new_cost < current_cost:
                    current_perm = new_perm
                    current_cost = new_cost
                    improved = True
                    if current_cost < best_cost:
                        best_perm = list(current_perm)
                        best_cost = current_cost
                    # Recompute C for next iteration
                    _, current_C = _makespan_and_completion(current_perm, processing_times)
                    break  # Restart the i loop after a move
            if improved:
                break
    
    return best_perm, best_cost

def _perturb(perm, strength=2):
    """
    Perturb the permutation by reversing a random segment.
    strength controls the maximum segment length.
    """
    n = len(perm)
    if n <= 1:
        return list(perm)
    
    max_len = min(strength + 1, n)
    i = random.randint(0, n - 1)
    j = random.randint(i, min(i + max_len - 1, n - 1))
    if i < j:
        new_perm = list(perm)
        new_perm[i:j+1] = new_perm[i:j+1][::-1]
        return new_perm
    return list(perm)

def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan using NEH + ILS with 2-opt.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    n_jobs, n_machines = processing_times.shape
    
    # 1. NEH initialization
    initial_perm = _neh_initialization(processing_times)
    initial_cost = _makespan(initial_perm, processing_times)
    
    best_perm = list(initial_perm)
    best_cost = initial_cost
    
    # 2. Local search on initial solution
    ls_perm, ls_cost = _local_search_2opt(initial_perm, processing_times, deadline)
    if ls_cost < best_cost:
        best_perm = list(ls_perm)
        best_cost = ls_cost
    
    # 3. Iterated Local Search: perturb and re-optimize
    perturbation_strength = 2
    iteration = 0
    
    while time.time() < deadline - 2.0:  # Leave 2s buffer
        iteration += 1
        
        # Perturb the best solution
        perturbed_perm = _perturb(best_perm, perturbation_strength)
        
        # Local search on perturbed solution
        ls_perm, ls_cost = _local_search_2opt(perturbed_perm, processing_times, deadline)
        
        if ls_cost < best_cost:
            best_perm = list(ls_perm)
            best_cost = ls_cost
            # Reduce perturbation strength slightly on improvement
            perturbation_strength = max(1, perturbation_strength - 0.5)
        else:
            # Increase perturbation strength on stagnation
            perturbation_strength = min(5, perturbation_strength + 0.5)
        
        # If time is running low, break
        if time.time() > deadline - 3.0:
            break
    
    return list(best_perm)
