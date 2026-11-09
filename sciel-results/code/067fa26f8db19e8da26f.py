"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(processing_times, order):
    """Compute makespan for a given job order efficiently using the recurrence."""
    if len(order) == 0:
        return 0.0
    n_machines = processing_times.shape[1]
    
    # c[j] holds the completion time on machine j for the current job in sequence
    c = np.zeros(n_machines)
    for job_idx in order:
        p_row = processing_times[job_idx]
        for j in range(n_machines):
            if j == 0:
                c[j] = c[j] + p_row[j]
            else:
                c[j] = max(c[j], c[j-1]) + p_row[j]
    return c[-1]


def _insertion_delta(processing_times, order, i, j):
    """
    Compute the change in makespan when moving job at position i to position j.
    order is a list of job indices.
    i is the source position, j is the destination position (after removal).
    We assume i < j for now (moving right). For i > j, we can swap roles or handle separately.
    Actually, let's just compute the makespan of the new sequence directly for simplicity in correctness,
    but that's O(n*m). For large n, we need a faster method.
    
    However, given the 150s limit and typical instance sizes in these synthetic tasks,
    a full makespan calculation per move might be acceptable if we limit the search space.
    
    Let's try a hybrid: use full makespan calculation but with early termination or limited neighborhood.
    Or, implement the O(m) delta calculation for insertion.
    
    Let's implement the O(m) delta calculation for moving a job from position i to j.
    
    Let the sequence be S.
    Move job at i to j.
    
    If i < j:
      The job is removed from i, inserted at j.
      The subsequence from i to j-1 shifts left.
      The job is inserted at j.
      
    This is complex. Let's stick to a robust 2-opt / swap local search with full makespan evaluation,
    but optimize the makespan calculation using NumPy where possible or just tight loops.
    
    Actually, the recurrence is O(n*m). If n=100, m=10, that's 1000 ops.
    If we do 1000 moves, that's 1e6 ops. Very fast.
    If n=1000, m=10, that's 10000 ops per makespan.
    1000 moves -> 1e7 ops. Still fast in Python? Maybe 0.1s.
    
    Let's use full makespan evaluation for simplicity and correctness, and rely on the time limit to stop.
    """
    # We'll just compute the new makespan directly in the search loop to avoid complex delta bugs.
    pass


def _neh_rule(processing_times):
    """NEH heuristic: sort by total processing time descending, then insert each job in best position."""
    n_jobs = processing_times.shape[0]
    totals = processing_times.sum(axis=1)
    order = list(np.argsort(-totals))
    
    # Start with first job
    seq = [order[0]]
    
    for i in range(1, n_jobs):
        job = order[i]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            # Try inserting job at pos
            new_seq = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(processing_times, new_seq)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq.insert(best_pos, job)
        
    return seq


def _local_search(processing_times, order, time_limit=10):
    """Swap-based local search. Returns improved order."""
    n_jobs = len(order)
    current_ms = _makespan(processing_times, order)
    best_order = list(order)
    best_ms = current_ms
    
    start_time = time.time()
    no_improve_count = 0
    max_no_improve = 1000  # Stop if no improvement for this many iterations
    
    while True:
        if time.time() - start_time > time_limit:
            break
        if no_improve_count > max_no_improve:
            break
            
        improved = False
        # Randomized swap neighborhood
        # To avoid getting stuck, we can shuffle the order of checks or use random pairs
        for i in range(n_jobs):
            for j in range(i + 1, n_jobs):
                # Swap i and j
                order[i], order[j] = order[j], order[i]
                new_ms = _makespan(processing_times, order)
                if new_ms < best_ms - 1e-9:
                    best_ms = new_ms
                    best_order = list(order)
                    improved = True
                    no_improve_count = 0
                else:
                    # Revert
                    order[i], order[j] = order[j], order[i]
                    
        if not improved:
            no_improve_count += 1
            # If no improvement in a full pass, we are at a local optimum for swaps.
            # Break to save time or try a different neighborhood?
            # For now, break.
            break
            
    return best_order


def order_jobs(processing_times):
    # Deadline guard
    start_time = time.time()
    time_limit = 140.0  # Leave 10s margin
    
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    # 1. Generate initial solution using NEH
    best_order = _neh_rule(processing_times)
    best_ms = _makespan(processing_times, best_order)
    
    # 2. Local search on the initial solution
    # Use a time budget for local search
    ls_time_budget = min(10.0, time_limit - (time.time() - start_time))
    if ls_time_budget > 1.0:
        improved_order = _local_search(processing_times, best_order, time_limit=ls_time_budget)
        improved_ms = _makespan(processing_times, improved_order)
        if improved_ms < best_ms:
            best_order = improved_order
            best_ms = improved_ms
    
    # 3. If time remains, try multiple random restarts with NEH + Local Search
    # NEH is deterministic, so we need to randomize it or use different heuristics.
    # We can randomize the tie-breaking in NEH or use a random permutation + local search.
    
    remaining_time = time_limit - (time.time() - start_time)
    if remaining_time > 2.0:
        # Try random restarts
        # Use a fixed number of restarts based on time
        # Each restart: random perm + local search (short time)
        # Or better: perturb the current best order and local search
        
        rng = np.random.default_rng(42)
        
        while time.time() - start_time < time_limit - 2.0:
            # Perturb the best order: swap 2-3 random pairs
            candidate = list(best_order)
            # Random swaps
            for _ in range(3):
                i, j = rng.integers(0, n_jobs, size=2)
                if i != j:
                    candidate[i], candidate[j] = candidate[j], candidate[i]
            
            # Local search with short time limit
            ls_time = min(2.0, time_limit - (time.time() - start_time) - 1.0)
            if ls_time < 0.5:
                break
                
            improved_candidate = _local_search(processing_times, candidate, time_limit=ls_time)
            candidate_ms = _makespan(processing_times, improved_candidate)
            
            if candidate_ms < best_ms - 1e-9:
                best_ms = candidate_ms
                best_order = improved_candidate
            else:
                # No improvement, try again
                continue
                
    return list(best_order)
# EVOLVE-BLOCK-END
