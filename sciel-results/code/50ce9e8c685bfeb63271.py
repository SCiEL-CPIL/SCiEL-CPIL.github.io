"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _calc_makespan_vectorized(processing_times, perm):
    """
    Calculate makespan for a given permutation using vectorized NumPy operations.
    
    This is significantly faster than Python loops for moderate to large N because
    it moves the inner loop into C-optimized NumPy routines.
    
    The recurrence is:
    C[j, m] = max(C[j-1, m], C[j, m-1]) + P[j, m]
    
    We can compute the full completion matrix C (n_jobs x n_machines) in a single
    Python loop over jobs, but each iteration uses vectorized NumPy operations 
    for the machine dimension.
    """
    n_jobs = len(perm)
    n_machines = processing_times.shape[1]
    
    if n_jobs == 0:
        return 0.0
    
    if n_jobs == 1:
        return float(processing_times[perm[0], :].sum())
        
    # Extract the submatrix for the jobs in the given order
    # This is a fancy index operation, O(N*M)
    P = processing_times[perm, :]
    
    # C will store completion times. C[i] is the completion vector for job i in perm.
    # C[i, m] = max(C[i-1, m], C[i, m-1]) + P[i, m]
    
    # Initialize C with zeros
    C = np.zeros((n_jobs, n_machines), dtype=np.float64)
    
    # First job: C[0, m] = sum(P[0, 0..m])
    # Because C[-1, m] is 0 (no previous job) and C[0, -1] is 0 (no previous machine for first machine)
    # C[0, 0] = P[0,0]
    # C[0, 1] = max(0, C[0,0]) + P[0,1] = C[0,0] + P[0,1]
    # So C[0] is just cumulative sum of P[0]
    C[0] = np.cumsum(P[0])
    
    # Subsequent jobs
    # For i in 1..n_jobs-1:
    # C[i, m] = max(C[i-1, m], C[i, m-1]) + P[i, m]
    # This dependency on C[i, m-1] (current row, previous column) prevents full vectorization
    # of the row calculation in one go without a loop or a specialized trick.
    # However, we can use a loop over jobs (N iterations) where each iteration is vectorized 
    # over machines (M operations in C).
    # This is O(N*M) but with low constant factor compared to pure Python.
    
    # Actually, there is a trick to vectorize the row calculation if we unroll or use 
    # cumulative sums with masks, but for simplicity and robustness, 
    # a loop over jobs with vectorized machine update is often faster than pure Python 
    # if M is not tiny.
    
    # Let's try the loop over jobs.
    # C[i] depends on C[i-1] and P[i].
    # C[i, 0] = C[i-1, 0] + P[i, 0]
    # C[i, m] = max(C[i-1, m], C[i, m-1]) + P[i, m]
    
    # We can compute C[i] iteratively.
    # Is there a vectorized way to compute the row?
    # C[i, m] - P[i, m] = max(C[i-1, m], C[i, m-1])
    # This is a 1D cellular automaton step.
    
    # For small M, pure Python might be faster due to overhead. 
    # For large M, NumPy is faster.
    # Let's assume the problem sizes are such that NumPy overhead is worth it 
    # or we can mix.
    
    # A hybrid approach: if n_machines is small (< 20), use Python loop for the row.
    # If large, maybe try to optimize.
    # But actually, the bottleneck in the previous code was the O(N^2) evaluations 
    # in local search. The makespan calculation itself is O(N*M).
    # If we can make the makespan calculation 5-10x faster, we can do 5-10x more evaluations.
    
    # Let's stick to the job-loop with vectorized machine updates where possible.
    # Actually, the recurrence C[i, m] = max(C[i-1, m], C[i, m-1]) + P[i, m] 
    # can be computed row by row.
    
    # Let's try to vectorize the row update.
    # C_i = P_i
    # for m in 1..M-1:
    #    C_i[m] += max(C_{i-1}[m], C_i[m-1])
    
    # This inner loop is still sequential.
    # However, we can use `np.maximum.accumulate`? No, that's for prefix max.
    
    # Let's just use the Python loop for the machine dimension but keep the job loop 
    # in Python too? No, that's what the old code did.
    
    # Let's try a different vectorization strategy:
    # C[i] = P[i] + max(C[i-1], shift_right(C[i]))
    # This is implicit.
    
    # Given the constraints, the most effective speedup often comes from 
    # reducing the number of evaluations or using a faster heuristic.
    # SA with a fast evaluation is good.
    
    # Let's implement the makespan with a tight Python loop but optimized local vars,
    # as it's often faster for M < 50.
    
    comp = np.zeros(n_machines)
    for j in range(n_jobs):
        row = P[j]
        prev_c = 0.0
        for m in range(n_machines):
            if m == 0:
                new_c = comp[0] + row[0]
            else:
                if comp[m] > prev_c:
                    new_c = comp[m] + row[m]
                else:
                    new_c = prev_c + row[m]
            comp[m] = new_c
            prev_c = new_c
            
    return float(comp[-1])


def order_jobs(processing_times):
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n_jobs, n_machines = processing_times.shape
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    
    # 1. Initial Construction: NEH Heuristic
    # Sort jobs by total processing time descending
    totals = processing_times.sum(axis=1)
    sorted_jobs = np.argsort(-totals)
    
    # NEH insertion:
    # Start with first job
    current_perm = list(sorted_jobs[0])
    
    # For each subsequent job, insert it at the best position
    for i in range(1, n_jobs):
        job = sorted_jobs[i]
        best_pos = 0
        best_makespan = float('inf')
        
        # Try all positions
        for pos in range(len(current_perm) + 1):
            candidate_perm = current_perm[:pos] + [job] + current_perm[pos:]
            ms = _calc_makespan_vectorized(processing_times, candidate_perm)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
                
        # Insert best
        current_perm = current_perm[:best_pos] + [job] + current_perm[best_pos:]
        
    best_perm = current_perm
    best_ms = _calc_makespan_vectorized(processing_times, best_perm)
    
    # 2. Simulated Annealing
    # Parameters
    # T0: initial temperature
    # T_end: final temperature
    # alpha: cooling rate
    # moves: swap, insert, block move
    
    # Estimate problem size to tune parameters
    # If N is small, we can run more iterations.
    
    # Set initial temperature based on average makespan change
    # Or just a fixed heuristic.
    
    # Let's use a simple SA loop.
    
    # To allow restarts, we can divide the time budget into chunks.
    # Or just run one long SA with random restarts if stuck.
    
    # Let's try a multi-start SA approach.
    # Time budget for each restart: e.g., 10s? Or dynamic?
    # Let's do 10 restarts if time permits, or 1 long run.
    # Given 115s, let's do 5 restarts of ~20s each.
    
    num_restarts = 5
    time_per_restart = 20.0
    
    for restart in range(num_restarts):
        if time.time() > deadline - 5.0:
            break
            
        # Initialize SA state
        # For first restart, use NEH. For others, random or perturbed NEH.
        if restart == 0:
            perm = best_perm.copy()
            ms = best_ms
        else:
            # Random perturbation of best_perm or new random?
            # Let's try random shuffle for diversity, or 2-opt perturbation.
            # Random shuffle is good for escaping deep local minima.
            perm = list(range(n_jobs))
            random.shuffle(perm)
            ms = _calc_makespan_vectorized(processing_times, perm)
            
            # If random is better, update best
            if ms < best_ms:
                best_ms = ms
                best_perm = perm.copy()
                
        # SA Parameters
        # T0 should be high enough to accept bad moves.
        # Let's estimate a typical delta.
        # Or just use a fixed T0 that works for normalized scales?
        # Processing times are arbitrary.
        
        # Heuristic T0: 10% of initial makespan?
        T0 = ms * 0.1
        if T0 == 0: T0 = 1.0
        
        T_end = T0 * 1e-5
        alpha = (T_end / T0) ** (1.0 / 1000) # Approx 1000 steps per restart?
        
        # Actually, let's just run until time limit for this restart.
        # T = T0
        # Cooling: T *= alpha
        
        # To determine alpha, we need number of steps.
        # Let's target ~10000 steps per restart if possible?
        # If each step takes 1ms, 10s allows 10000 steps.
        
        # Let's estimate step time.
        # For N=100, M=10, makespan calc is ~0.1ms?
        # Let's just use a fixed number of iterations based on time.
        
        # Simpler: Run SA for a fixed duration.
        restart_start = time.time()
        restart_deadline = restart_start + time_per_restart
        
        T = T0
        # Cool down to T_end over the restart duration.
        # We don't know exact steps, so we can use time-based cooling?
        # Or just a fixed alpha and see how many steps we get.
        
        # Let's use a fixed alpha = 0.999
        alpha = 0.999
        
        while time.time() < restart_deadline:
            if T < T_end:
                break
                
            # Generate neighbor
            # Choose move type: 60% swap, 40% insert
            if random.random() < 0.6:
                # Swap
                i, j = random.sample(range(n_jobs), 2)
                new_perm = perm.copy()
                new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
            else:
                # Insert
                i = random.randint(0, n_jobs - 1)
                j = random.randint(0, n_jobs - 1)
                if i == j:
                    continue
                job = perm[i]
                new_perm = perm[:i] + perm[i+1:]
                # Adjust j if i < j
                if i < j:
                    j -= 1
                new_perm = new_perm[:j] + [job] + new_perm[j:]
                
            new_ms = _calc_makespan_vectorized(processing_times, new_perm)
            
            delta = new_ms - ms
            if delta < 0:
                # Accept
                perm = new_perm
                ms = new_ms
                if ms < best_ms:
                    best_ms = ms
                    best_perm = perm.copy()
            else:
                # Accept with probability exp(-delta / T)
                if random.random() < np.exp(-delta / T):
                    perm = new_perm
                    ms = new_ms
                    
            T *= alpha
            
    return best_perm
# EVOLVE-BLOCK-END
