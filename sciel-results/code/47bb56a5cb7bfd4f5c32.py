"""Flow-shop scheduling with ACO + local search."""
import numpy as np
import time
from scipy.special import comb

# EVOLVE-BLOCK-START
import numpy as np
import time


def _makespan(perm, processing_times):
    """Compute makespan for a given permutation.
    
    Args:
        perm: list/array of job indices in order
        processing_times: np.ndarray of shape (n_jobs, n_machines)
    
    Returns:
        float makespan
    """
    n_machines = processing_times.shape[1]
    # completion_time[m] = completion time on machine m
    completion_time = np.zeros(n_machines)
    for job in perm:
        for m in range(n_machines):
            if m == 0:
                completion_time[m] += processing_times[job, m]
            else:
                completion_time[m] = max(completion_time[m], completion_time[m - 1]) + processing_times[job, m]
    return completion_time[-1]


def _makespan_fast(perm, processing_times):
    """Faster makespan using vectorized operations where possible."""
    n_machines = processing_times.shape[1]
    completion_time = np.zeros(n_machines)
    for job in perm:
        # Vectorized update
        if n_machines > 1:
            completion_time[1:] = np.maximum(completion_time[1:], completion_time[:-1]) + processing_times[job, 1:]
        completion_time[0] += processing_times[job, 0]
    return completion_time[-1]


def _neh_heuristic(processing_times):
    """NEH (Nawaz-Enscore-Ham) heuristic for initial solution."""
    n_jobs, n_machines = processing_times.shape
    
    # Sort jobs by total processing time in descending order
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    best_perm = list(sorted_jobs)
    best_makespan = _makespan_fast(best_perm, processing_times)
    
    # Try inserting each job into all positions
    for i in range(1, n_jobs):
        job = sorted_jobs[i]
        best_pos = 0
        best_ms = best_makespan
        
        current_perm = best_perm[:]
        # Remove job from current position
        current_perm.pop(i)
        
        for pos in range(len(current_perm) + 1):
            test_perm = current_perm[:pos] + [job] + current_perm[pos:]
            ms = _makespan_fast(test_perm, processing_times)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        
        # Reinsert job at best position
        best_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
        best_makespan = best_ms
    
    return best_perm, best_makespan


def _local_search(perm, processing_times, max_iterations=200, time_limit=5.0):
    """Local search with swap and insert moves."""
    n_jobs = len(perm)
    best_perm = list(perm)
    best_makespan = _makespan_fast(best_perm, processing_times)
    start_time = time.time()
    iteration = 0
    
    while iteration < max_iterations and (time.time() - start_time) < time_limit:
        improved = False
        iteration += 1
        
        # Try swap moves
        for i in range(n_jobs):
            for j in range(i + 1, n_jobs):
                test_perm = best_perm[:]
                test_perm[i], test_perm[j] = test_perm[j], test_perm[i]
                ms = _makespan_fast(test_perm, processing_times)
                if ms < best_makespan:
                    best_perm = test_perm
                    best_makespan = ms
                    improved = True
        
        # Try insert moves
        for i in range(n_jobs):
            for j in range(n_jobs):
                if i == j or i == j + 1:
                    continue
                test_perm = best_perm[:]
                job = test_perm.pop(i)
                if j > i:
                    j -= 1
                test_perm.insert(j, job)
                ms = _makespan_fast(test_perm, processing_times)
                if ms < best_makespan:
                    best_perm = test_perm
                    best_makespan = ms
                    improved = True
        
        if not improved:
            break
    
    return best_perm, best_makespan


def _aco_iteration(pheromone, processing_times, alpha, beta, n_ants, time_limit=5.0):
    """One ACO iteration: construct solutions using pheromone and heuristic info."""
    n_jobs, n_machines = processing_times.shape
    start_time = time.time()
    
    # Precompute heuristic values (1/total processing time for each job)
    totals = processing_times.sum(axis=1)
    heuristic = 1.0 / (totals + 1e-10)
    
    solutions = []
    makespans = []
    
    for ant in range(n_ants):
        if (time.time() - start_time) > time_limit:
            break
        
        # Construct solution
        remaining_jobs = set(range(n_jobs))
        perm = []
        
        for _ in range(n_jobs):
            if not remaining_jobs:
                break
            
            # Calculate probabilities
            probs = np.zeros(len(remaining_jobs))
            for idx, job in enumerate(remaining_jobs):
                if len(perm) == 0:
                    # No previous job, use heuristic only
                    probs[idx] = heuristic[job] ** beta
                else:
                    prev_job = perm[-1]
                    pher = pheromone[prev_job, job]
                    probs[idx] = (pher ** alpha) * (heuristic[job] ** beta)
            
            total_prob = probs.sum()
            if total_prob < 1e-10:
                # Fallback to random
                job = list(remaining_jobs)[np.random.randint(len(remaining_jobs))]
            else:
                probs = probs / total_prob
                # Roulette wheel selection
                r = np.random.random()
                cumulative = np.cumsum(probs)
                idx = np.searchsorted(cumulative, r)
                if idx >= len(remaining_jobs):
                    idx = len(remaining_jobs) - 1
                job = list(remaining_jobs)[idx]
            
            perm.append(job)
            remaining_jobs.remove(job)
        
        ms = _makespan_fast(perm, processing_times)
        solutions.append(perm)
        makespans.append(ms)
    
    return solutions, makespans


def _update_pheromone(pheromone, solutions, makespans, rho, Q):
    """Update pheromone trails."""
    n_jobs = pheromone.shape[0]
    
    # Evaporation
    pheromone *= (1.0 - rho)
    
    # Add pheromone based on solution quality
    for perm, ms in zip(solutions, makespans):
        if ms > 0:
            deposit = Q / ms
            
            # Deposit on edges
            for i in range(len(perm) - 1):
                prev_job = perm[i]
                curr_job = perm[i + 1]
                pheromone[prev_job, curr_job] += deposit
                # Also deposit in reverse for undirected graph
                pheromone[curr_job, prev_job] += deposit * 0.5  # Half weight for reverse
    
    return pheromone


def order_jobs(processing_times):
    """Main entry point: find optimal job permutation using ACO + local search."""
    n_jobs, n_machines = processing_times.shape
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s limit
    
    # Get initial solution using NEH heuristic
    initial_perm, initial_makespan = _neh_heuristic(processing_times)
    
    # Apply local search to initial solution
    best_perm, best_makespan = _local_search(initial_perm, processing_times, 
                                               max_iterations=50, time_limit=3.0)
    
    # Initialize pheromone matrix
    pheromone = np.ones((n_jobs, n_jobs)) * 0.1
    np.fill_diagonal(pheromone, 0.0)
    
    # ACO parameters
    alpha = 1.0  # Pheromone importance
    beta = 2.0   # Heuristic importance
    rho = 0.5    # Evaporation rate
    Q = 1.0      # Pheromone constant
    n_ants = 10  # Number of ants per iteration
    
    # Run ACO iterations
    while (time.time() - start_time) < deadline:
        # Check if we have enough time for one more iteration
        if (time.time() - start_time) > deadline - 5.0:
            break
        
        # Construct solutions
        solutions, makespans = _aco_iteration(pheromone, processing_times, 
                                               alpha, beta, n_ants, time_limit=3.0)
        
        if not solutions:
            break
        
        # Find best solution in this iteration
        best_idx = np.argmin(makespans)
        iter_best_perm = solutions[best_idx]
        iter_best_makespan = makespans[best_idx]
        
        # Apply local search to the best solution from this iteration
        ls_perm, ls_makespan = _local_search(iter_best_perm, processing_times,
                                              max_iterations=30, time_limit=2.0)
        
        if ls_makespan < best_makespan:
            best_perm = ls_perm
            best_makespan = ls_makespan
        
        # Update pheromone
        pheromone = _update_pheromone(pheromone, solutions, makespans, rho, Q)
        
        # Normalize pheromone to prevent numerical issues
        max_pher = pheromone.max()
        if max_pher > 0:
            pheromone /= max_pher
            pheromone *= 0.5  # Scale down to keep values reasonable
    
    # Final local search on the best solution found
    if (time.time() - start_time) < deadline - 2.0:
        best_perm, best_makespan = _local_search(best_perm, processing_times,
                                                  max_iterations=100, time_limit=5.0)
    
    return list(best_perm)
# EVOLVE-BLOCK-END
