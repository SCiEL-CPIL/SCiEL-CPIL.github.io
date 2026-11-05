"""Improved flow-shop scheduling via NEH heuristic, fast makespan, insertion local search, and simulated annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(pt, perm):
    """Compute makespan for a given permutation efficiently."""
    n_jobs, n_machines = pt.shape
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    
    # Use Python lists for speed
    prev = [0.0] * n_machines
    for j in perm:
        curr = [0.0] * n_machines
        curr[0] = prev[0] + pt[j, 0]
        for m in range(1, n_machines):
            val = prev[m] if prev[m] > curr[m-1] else curr[m-1]
            curr[m] = val + pt[j, m]
        prev = curr
    return prev[-1]


def _neh_heuristic(processing_times):
    """NEH heuristic for flow shop scheduling."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    # Insert each job into the best position in the sequence
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


def _insertion_local_search(processing_times, initial_order, time_limit=10.0):
    """Insertion-based local search with time limit."""
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
        # Try all insertion moves: remove job at position i and insert at position j
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            
            job = current_order[i]
            # Remove job from current position
            remaining = current_order[:i] + current_order[i+1:]
            
            for j in range(len(remaining) + 1):
                if time.time() - start_time > time_limit:
                    break
                
                # Skip if inserting back at same position
                if j == i or (j == i - 1 and i > 0):
                    continue
                    
                candidate = remaining[:j] + [job] + remaining[j:]
                new_makespan = _makespan(processing_times, candidate)
                
                if new_makespan < current_makespan - 1e-9:
                    current_order = candidate
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


def _swap_local_search(processing_times, initial_order, time_limit=10.0):
    """Swap-based local search with time limit."""
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
        # Try all swap moves
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


def _simulated_annealing(processing_times, initial_order, time_limit=15.0):
    """Simulated annealing with insertion neighborhood."""
    n_jobs = len(initial_order)
    if n_jobs <= 1:
        return list(initial_order)
    
    current_order = list(initial_order)
    current_makespan = _makespan(processing_times, current_order)
    best_order = list(current_order)
    best_makespan = current_makespan
    
    start_time = time.time()
    deadline = start_time + time_limit
    
    # Temperature schedule
    T = max(current_makespan * 0.1, 1.0)
    cooling_rate = 0.995
    
    np.random.seed(42)
    
    while time.time() < deadline:
        # Generate a random insertion move
        i = np.random.randint(n_jobs)
        j = np.random.randint(n_jobs)
        if i == j:
            continue
            
        job = current_order[i]
        remaining = current_order[:i] + current_order[i+1:]
        
        # Adjust j if necessary
        if j > i:
            j -= 1
        if j < 0:
            j = 0
        if j > len(remaining):
            j = len(remaining)
            
        candidate = remaining[:j] + [job] + remaining[j:]
        new_makespan = _makespan(processing_times, candidate)
        
        delta = new_makespan - current_makespan
        
        if delta < 0 or np.random.random() < np.exp(-delta / T):
            current_order = candidate
            current_makespan = new_makespan
            
            if current_makespan < best_makespan:
                best_makespan = current_makespan
                best_order = list(current_order)
        
        # Cool down
        T *= cooling_rate
        if T < 1e-6:
            T = 1e-6
    
    return best_order


def _random_perturbation(order, strength=0.3):
    """Apply random insertion perturbation."""
    n = len(order)
    if n <= 1:
        return list(order)
    
    new_order = list(order)
    num_moves = max(1, int(strength * n))
    
    for _ in range(num_moves):
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            continue
        job = new_order[i]
        remaining = new_order[:i] + new_order[i+1:]
        
        if j > i:
            j -= 1
        if j < 0:
            j = 0
        if j > len(remaining):
            j = len(remaining)
            
        new_order = remaining[:j] + [job] + remaining[j:]
    
    return new_order


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow shop.
    Uses NEH heuristic followed by multiple local search strategies and simulated annealing.
    """
    processing_times = np.asarray(processing_times, dtype=np.float64)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Deadline guard: we must return well before 150s
    deadline = 140.0  # 10s safety margin
    start_time = time.time()
    
    # Step 1: NEH heuristic
    neh_order = _neh_heuristic(processing_times)
    neh_makespan = _makespan(processing_times, neh_order)
    
    best_order = list(neh_order)
    best_makespan = neh_makespan
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if remaining < 5.0:
        return best_order
    
    # Step 2: Insertion local search on the NEH solution
    ls_time = min(remaining * 0.3, 30.0)
    ls_order = _insertion_local_search(processing_times, best_order, time_limit=ls_time)
    ls_makespan = _makespan(processing_times, ls_order)
    
    if ls_makespan < best_makespan:
        best_order = list(ls_order)
        best_makespan = ls_makespan
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if remaining < 5.0:
        return best_order
    
    # Step 3: Swap local search
    ls_time = min(remaining * 0.3, 30.0)
    ls_order = _swap_local_search(processing_times, best_order, time_limit=ls_time)
    ls_makespan = _makespan(processing_times, ls_order)
    
    if ls_makespan < best_makespan:
        best_order = list(ls_order)
        best_makespan = ls_makespan
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if remaining < 5.0:
        return best_order
    
    # Step 4: Simulated annealing
    sa_time = min(remaining * 0.4, 40.0)
    sa_order = _simulated_annealing(processing_times, best_order, time_limit=sa_time)
    sa_makespan = _makespan(processing_times, sa_order)
    
    if sa_makespan < best_makespan:
        best_order = list(sa_order)
        best_makespan = sa_makespan
    
    elapsed = time.time() - start_time
    remaining = deadline - elapsed
    
    if remaining < 3.0:
        return best_order
    
    # Step 5: Multiple restarts with perturbation
    np.random.seed(42)
    num_restarts = min(10, int(remaining / 3.0))
    
    for r in range(num_restarts):
        if time.time() + 3.0 > deadline:
            break
        
        # Perturb the best solution
        perturbed = _random_perturbation(best_order, strength=0.2 + 0.05 * r)
        
        # Local search on perturbed solution
        ls_time_r = min(2.0, deadline - time.time() - 1.0)
        if ls_time_r < 0.1:
            break
        
        ls_order_r = _insertion_local_search(processing_times, perturbed, time_limit=ls_time_r)
        ls_makespan_r = _makespan(processing_times, ls_order_r)
        
        if ls_makespan_r < best_makespan:
            best_order = list(ls_order_r)
            best_makespan = ls_makespan_r
    
    return best_order
# EVOLVE-BLOCK-END
