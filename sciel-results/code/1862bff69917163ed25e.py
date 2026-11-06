"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(order, processing_times):
    """Compute makespan for a given job order."""
    n_machines = processing_times.shape[1]
    if n_machines == 0 or len(order) == 0:
        return 0.0
    completion = np.zeros(n_machines)
    for job in order:
        p = processing_times[job]
        prev = 0.0
        for m in range(n_machines):
            prev = max(prev, completion[m]) + p[m]
            completion[m] = prev
    return float(completion[-1])


def _makespan_vectorized(order, processing_times):
    """Faster makespan computation using NumPy where possible."""
    n_machines = processing_times.shape[1]
    n_jobs = len(order)
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    
    seq = np.asarray(order, dtype=np.int64)
    P = processing_times[seq]  # shape (n_jobs, n_machines)
    
    # C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m]
    # We can't fully vectorize due to the dependency, but we can use a tighter loop
    C = np.zeros((n_jobs, n_machines))
    for j in range(n_jobs):
        for m in range(n_machines):
            if j == 0 and m == 0:
                C[j, m] = P[j, m]
            elif j == 0:
                C[j, m] = C[j, m-1] + P[j, m]
            elif m == 0:
                C[j, m] = C[j-1, m] + P[j, m]
            else:
                C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m]
    
    return float(C[-1, -1])


def _neh_initialization(processing_times):
    """NEH heuristic for initial solution."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = list(np.argsort(-totals))
    
    # Build the sequence using NEH insertion
    seq = [sorted_jobs[0]]
    for job in sorted_jobs[1:]:
        # Try all possible insertion positions
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            candidate = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(candidate, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    
    return seq


def _improve_with_local_search(processing_times, initial_order, time_limit=10.0):
    """Improve an initial order using pairwise swap local search with time limit."""
    n_jobs = len(initial_order)
    if n_jobs <= 1:
        return initial_order
    
    start_time = time.time()
    current_order = list(initial_order)
    current_ms = _makespan(current_order, processing_times)
    best_order = list(current_order)
    best_ms = current_ms
    
    # Tabu-like: keep track of recent swaps to avoid cycling
    recent_swaps = set()
    max_recent = min(n_jobs * 2, 50)
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        improved = False
        # Try all pairwise swaps
        for i in range(n_jobs):
            if time.time() - start_time > time_limit:
                break
            for j in range(i + 1, n_jobs):
                if (i, j) in recent_swaps or (j, i) in recent_swaps:
                    continue
                
                # Perform swap
                current_order[i], current_order[j] = current_order[j], current_order[i]
                new_ms = _makespan(current_order, processing_times)
                
                if new_ms < current_ms - 1e-9:
                    # Accept improvement
                    current_ms = new_ms
                    if new_ms < best_ms - 1e-9:
                        best_ms = new_ms
                        best_order = list(current_order)
                    improved = True
                    # Add to recent swaps
                    recent_swaps.add((i, j))
                    recent_swaps.add((j, i))
                    if len(recent_swaps) > max_recent:
                        # Remove oldest entries
                        # Simple approach: clear half
                        it = iter(recent_swaps)
                        for _ in range(max_recent // 2):
                            recent_swaps.discard(next(it))
                else:
                    # Revert swap
                    current_order[i], current_order[j] = current_order[j], current_order[i]
        
        if not improved:
            break
    
    return best_order


def _simulated_annealing(processing_times, initial_order, time_limit=10.0):
    """Simulated annealing for further improvement."""
    n_jobs = len(initial_order)
    if n_jobs <= 1:
        return initial_order
    
    start_time = time.time()
    current_order = list(initial_order)
    current_ms = _makespan(current_order, processing_times)
    best_order = list(current_order)
    best_ms = current_ms
    
    # Temperature parameters
    T0 = max(1.0, current_ms * 0.1)
    T_min = 0.01
    cooling_rate = 0.95
    
    T = T0
    
    while T > T_min:
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Generate a random neighbor by swapping two random positions
        i, j = np.random.choice(n_jobs, size=2, replace=False)
        current_order[i], current_order[j] = current_order[j], current_order[i]
        new_ms = _makespan(current_order, processing_times)
        
        delta = new_ms - current_ms
        
        if delta < 0 or np.random.random() < np.exp(-delta / max(T, 1e-10)):
            # Accept
            current_ms = new_ms
            if new_ms < best_ms - 1e-9:
                best_ms = new_ms
                best_order = list(current_order)
        else:
            # Revert
            current_order[i], current_order[j] = current_order[j], current_order[i]
        
        T *= cooling_rate
    
    return best_order


def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on flow-shop instances."""
    n_jobs, n_machines = processing_times.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # Overall deadline guard
    overall_start = time.time()
    overall_limit = 110.0  # Leave 10s safety margin for 120s limit
    
    # Phase 1: Generate multiple initial solutions using different heuristics
    initial_solutions = []
    
    # NEH initialization
    neh_order = _neh_initialization(processing_times)
    initial_solutions.append(neh_order)
    
    # Longest processing time first
    lpt_order = list(np.argsort(-processing_times.sum(axis=1)))
    initial_solutions.append(lpt_order)
    
    # Random initializations
    n_random = min(5, max(1, (10000 // max(n_jobs, 1))))
    for _ in range(n_random):
        rand_order = list(np.random.permutation(n_jobs))
        initial_solutions.append(rand_order)
    
    # Phase 2: Improve each initial solution with local search
    improved_solutions = []
    per_solution_time = max(1.0, (overall_limit * 0.5) / len(initial_solutions))
    
    for sol in initial_solutions:
        if time.time() - overall_start > overall_limit * 0.6:
            break
        improved = _improve_with_local_search(processing_times, sol, time_limit=per_solution_time)
        improved_solutions.append(improved)
    
    # Phase 3: Further improve the best solution with simulated annealing
    if improved_solutions:
        # Find the best improved solution
        best_so_far = min(improved_solutions, key=lambda x: _makespan(x, processing_times))
        best_ms_so_far = _makespan(best_so_far, processing_times)
        
        # Use remaining time for simulated annealing
        elapsed = time.time() - overall_start
        sa_time = max(1.0, overall_limit - elapsed - 5.0)
        if sa_time > 1.0:
            sa_result = _simulated_annealing(processing_times, best_so_far, time_limit=sa_time)
            sa_ms = _makespan(sa_result, processing_times)
            if sa_ms < best_ms_so_far:
                best_so_far = sa_result
                best_ms_so_far = sa_ms
    
    # Final deadline check
    if time.time() - overall_start > overall_limit:
        # Return whatever we have
        if improved_solutions:
            return min(improved_solutions, key=lambda x: _makespan(x, processing_times))
        return initial_solutions[0]
    
    if improved_solutions:
        return min(improved_solutions, key=lambda x: _makespan(x, processing_times))
    return initial_solutions[0]
# EVOLVE-BLOCK-END
