"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(processing_times, perm):
    """Compute makespan for a given permutation."""
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    if n_jobs == 0:
        return 0.0
    
    # Compute cumulative completion times
    # For each job in order, compute completion on each machine
    # C[i][j] = max(C[i-1][j], C[i][j-1]) + P[i][j]
    # where i is job index in permutation, j is machine index
    
    # Vectorized approach:
    # completion[i, j] = max(completion[i, j-1], completion[i-1, j]) + P[perm[i], j]
    # completion[-1, j] = P[perm[0], j] for first job
    
    P = processing_times[perm]  # (n_jobs, n_machines)
    
    if n_machines == 1:
        return float(P.sum())
    
    # Use iterative computation
    # completion[i, j] = max(completion[i, j-1], completion[i-1, j]) + P[i, j]
    # We can compute this row by row
    
    # completion for first job
    comp = np.zeros((n_jobs, n_machines))
    comp[0, 0] = P[0, 0]
    for j in range(1, n_machines):
        comp[0, j] = comp[0, j-1] + P[0, j]
    
    for i in range(1, n_jobs):
        comp[i, 0] = comp[i-1, 0] + P[i, 0]
        for j in range(1, n_machines):
            comp[i, j] = max(comp[i-1, j], comp[i, j-1]) + P[i, j]
    
    return float(comp[-1, -1])


def _makespan_delta(processing_times, perm, i, j):
    """
    Compute the change in makespan if we swap positions i and j in perm.
    This is an approximation that only considers the affected region.
    For exact computation, we need to recompute from position min(i,j) onwards.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    if i == j:
        return 0.0
    
    # Compute current makespan
    current_ms = _makespan(processing_times, perm)
    
    # Create swapped permutation
    new_perm = list(perm)
    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
    
    # Compute new makespan
    new_ms = _makespan(processing_times, new_perm)
    
    return new_ms - current_ms


def _neh_initialization(processing_times):
    """
    NEH (Nawaz, Enscore, Ham) heuristic for initial solution.
    This is a well-known heuristic for flow shop scheduling.
    """
    n_jobs = processing_times.shape[0]
    n_machines = processing_times.shape[1]
    
    if n_jobs <= 2:
        return list(range(n_jobs))
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    perm = []
    for job in sorted_jobs:
        # Try all positions and pick the one with minimum makespan
        best_pos = 0
        best_ms = float('inf')
        
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        perm.insert(best_pos, job)
    
    return perm


def _improve_with_local_search(processing_times, perm, max_iter=1000, time_limit=110.0):
    """
    Improve the permutation using local search with 2-opt moves (adjacent and non-adjacent swaps).
    Uses simulated annealing to escape local optima.
    """
    start_time = time.time()
    n_jobs = len(perm)
    
    if n_jobs <= 2:
        return perm
    
    current_perm = list(perm)
    current_ms = _makespan(processing_times, current_perm)
    best_perm = list(current_perm)
    best_ms = current_ms
    
    # Simulated annealing parameters
    initial_temp = current_ms * 0.1 if current_ms > 0 else 1.0
    cooling_rate = 0.95
    min_temp = 1e-6
    
    temp = initial_temp
    no_improve_count = 0
    max_no_improve = 50
    
    import random
    rng = random.Random(42)
    
    while temp > min_temp:
        if time.time() - start_time > time_limit:
            break
        
        # Generate a random move
        if n_jobs > 1:
            if rng.random() < 0.5:
                # Adjacent swap
                i = rng.randint(0, n_jobs - 2)
                j = i + 1
            else:
                # Random swap
                i = rng.randint(0, n_jobs - 1)
                j = rng.randint(0, n_jobs - 1)
                if i == j:
                    continue
        else:
            break
        
        # Compute new makespan
        new_perm = list(current_perm)
        new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
        new_ms = _makespan(processing_times, new_perm)
        
        delta = new_ms - current_ms
        
        if delta < 0:
            # Accept improvement
            current_perm = new_perm
            current_ms = new_ms
            no_improve_count = 0
            
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = list(current_perm)
        else:
            # Accept with probability exp(-delta / temp)
            if delta > 0 and temp > 0:
                prob = np.exp(-delta / temp)
                if rng.random() < prob:
                    current_perm = new_perm
                    current_ms = new_ms
                    no_improve_count += 1
            else:
                no_improve_count += 1
        
        # Cool down
        temp *= cooling_rate
        
        if no_improve_count > max_no_improve:
            # Restart with best solution
            current_perm = list(best_perm)
            current_ms = best_ms
            temp = initial_temp * 0.5
            no_improve_count = 0
    
    return best_perm


def _multi_start(processing_times, n_starts=5, time_per_start=100.0):
    """
    Multi-start local search with different initializations.
    """
    start_time = time.time()
    n_jobs = processing_times.shape[0]
    
    if n_jobs <= 2:
        return list(range(n_jobs))
    
    best_perm = None
    best_ms = float('inf')
    
    # First, try NEH initialization
    neh_perm = _neh_initialization(processing_times)
    improved = _improve_with_local_search(processing_times, neh_perm, time_limit=time_per_start)
    ms = _makespan(processing_times, improved)
    
    if ms < best_ms:
        best_ms = ms
        best_perm = improved
    
    if time.time() - start_time > time_per_start * 0.7:
        return best_perm
    
    # Try other initializations
    for start_idx in range(n_starts - 1):
        if time.time() - start_time > time_per_start * 0.9:
            break
        
        # Random initialization with some structure
        perm = list(range(n_jobs))
        
        # Apply some random swaps to diversify
        n_swaps = n_jobs // 2
        for _ in range(n_swaps):
            i = np.random.randint(0, n_jobs)
            j = np.random.randint(0, n_jobs)
            perm[i], perm[j] = perm[j], perm[i]
        
        improved = _improve_with_local_search(processing_times, perm, time_limit=time_per_start / (n_starts - start_idx))
        ms = _makespan(processing_times, improved)
        
        if ms < best_ms:
            best_ms = ms
            best_perm = improved
    
    return best_perm


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan on a flow-shop instance.
    
    Args:
        processing_times: np.ndarray of shape (n_jobs, n_machines)
        
    Returns:
        List of job indices in the optimal (or near-optimal) order.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    processing_times = np.asarray(processing_times)
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 1:
        # For single machine, any order gives same makespan, but let's sort by processing time
        return list(np.argsort(processing_times[:, 0]))
    
    # Determine number of multi-starts based on time budget
    time_elapsed = time.time() - start_time
    remaining_time = deadline - time_elapsed
    
    if n_jobs <= 10:
        # Small problem: use more starts
        n_starts = min(8, max(3, int(remaining_time / 15)))
    elif n_jobs <= 30:
        n_starts = min(5, max(2, int(remaining_time / 20)))
    else:
        n_starts = min(3, max(1, int(remaining_time / 30)))
    
    # Use multi-start local search
    best_perm = _multi_start(processing_times, n_starts=n_starts, time_per_start=min(remaining_time * 0.9, 100.0))
    
    # Check if we're still within deadline
    if time.time() < deadline:
        # Do one final local search pass on the best solution
        final_improved = _improve_with_local_search(processing_times, best_perm, time_limit=deadline - time.time() - 1.0)
        if _makespan(processing_times, final_improved) < _makespan(processing_times, best_perm):
            best_perm = final_improved
    
    return list(best_perm)
# EVOLVE-BLOCK-END
