# EVOLVE-BLOCK-START
import numpy as np
import time

def _makespan(perm, pt):
    """Compute makespan for a given permutation using the flow-shop recurrence.
    pt: (n, m) array of processing times.
    perm: array of job indices in order.
    Returns makespan (scalar).
    """
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    comp = np.zeros(m, dtype=float)
    for j in range(n):
        job = perm[j]
        pt_row = pt[job]
        for mi in range(m):
            prev_machine = comp[mi - 1] if mi > 0 else 0.0
            same_machine_prev = comp[mi]
            comp[mi] = max(same_machine_prev, prev_machine) + pt_row[mi]
    return comp[-1]

def _compute_completion_matrix(perm, pt):
    """Compute completion times for all machines for a given permutation.
    Returns (n, m) array where C[j, mi] is completion time of job j on machine mi."""
    n = len(perm)
    if n == 0:
        return np.zeros((0, pt.shape[1]), dtype=float)
    m = pt.shape[1]
    C = np.zeros((n, m), dtype=float)
    for j in range(n):
        job = perm[j]
        pt_row = pt[job]
        for mi in range(m):
            if mi == 0:
                if j == 0:
                    C[j, mi] = pt_row[mi]
                else:
                    C[j, mi] = C[j-1, mi] + pt_row[mi]
            else:
                if j == 0:
                    C[j, mi] = C[j, mi-1] + pt_row[mi]
                else:
                    C[j, mi] = max(C[j-1, mi], C[j, mi-1]) + pt_row[mi]
    return C

def _makespan_from_completion(C):
    """Extract makespan from completion matrix."""
    if C.size == 0:
        return 0.0
    return C[-1, -1]

def _neh_initial(pt):
    """NEH heuristic: sort jobs by decreasing total processing time, then insert
    each job at the position that minimizes makespan of the partial schedule."""
    n, m = pt.shape
    if n == 0:
        return np.array([], dtype=int)
    
    # Sort by decreasing total processing time
    total_pt = pt.sum(axis=1)
    order = np.argsort(-total_pt).astype(int)
    
    # Build schedule by inserting each job at the best position
    schedule = []
    for job in order:
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(schedule) + 1):
            new_schedule = schedule[:pos] + [job] + schedule[pos:]
            ms = _makespan(np.array(new_schedule), pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        schedule.insert(best_pos, job)
    
    return np.array(schedule, dtype=int)

def _palmer_initial(pt):
    """Palmer's heuristic: sort by decreasing key."""
    n, m = pt.shape
    if n == 0:
        return np.array([], dtype=int)
    
    keys = np.zeros(n)
    for j in range(n):
        for mi in range(m):
            keys[j] += (n - 1 - 2 * mi) * pt[j, mi]
    return np.argsort(-keys).astype(int)

def _bottleneck_initial(pt):
    """Sort by decreasing processing time on the bottleneck machine."""
    n, m = pt.shape
    if m == 0:
        return np.arange(n)
    machine_totals = pt.sum(axis=0)
    bottleneck_machine = np.argmax(machine_totals)
    return np.argsort(-pt[:, bottleneck_machine]).astype(int)

def _random_initial(n, rng):
    """Random permutation."""
    return rng.permutation(n)

def _pfs_decomposition_construct(pt, rng, num_constructs=5):
    """PFS-specific decomposition: fix first job (highest total PT), treat rest as TSP.
    Uses pre-computed incremental makespan matrix for efficient construction."""
    n, m = pt.shape
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)
    
    # Fix first job: job with highest total processing time
    total_pt = pt.sum(axis=1)
    first_job = np.argmax(total_pt)
    
    # Remaining jobs
    remaining_jobs = [j for j in range(n) if j != first_job]
    n_rem = len(remaining_jobs)
    
    if n_rem == 0:
        return np.array([first_job], dtype=int)
    
    # Pre-compute incremental makespan contribution matrix
    # delta[i][j] = additional makespan if job j is placed after job i in the sequence
    # This is an approximation: we compute it based on the completion times
    # For efficiency, we'll use a simpler heuristic: sum of processing times weighted by position
    
    # Better approach: use the actual flow-shop recurrence incrementally
    # delta[i][j] represents the cost of having job j immediately after job i
    # We approximate this by the makespan contribution when j follows i
    
    # Compute pairwise cost matrix for the remaining jobs
    # cost[i][j] = incremental makespan when job j is placed after job i
    # This requires computing completion times, which is expensive
    # Instead, use a simpler surrogate: sum of processing times on critical path
    
    # Use Palmer-like key for pairwise ordering
    # Or use the actual incremental computation for small n
    
    if n_rem <= 30:
        # Compute exact incremental cost matrix
        delta = np.zeros((n_rem, n_rem))
        for i_idx, i_job in enumerate(remaining_jobs):
            for j_idx, j_job in enumerate(remaining_jobs):
                if i_idx == j_idx:
                    continue
                # Compute makespan if j follows i (with first_job at start)
                test_perm = np.array([first_job, i_job, j_job])
                # Extend with other jobs in some order to get full makespan
                # For simplicity, just use the two-job extension
                ms = _makespan(test_perm, pt)
                delta[i_idx, j_idx] = ms
        # Normalize: subtract baseline
        baseline = np.mean(delta)
        delta -= baseline
    else:
        # For larger n, use a faster approximation
        # Use processing time on bottleneck machine as proxy
        machine_totals = pt.sum(axis=0)
        bottleneck_machine = np.argmax(machine_totals)
        delta = np.zeros((n_rem, n_rem))
        for i_idx, i_job in enumerate(remaining_jobs):
            for j_idx, j_job in enumerate(remaining_jobs):
                if i_idx == j_idx:
                    continue
                # Cost based on processing times on bottleneck machine
                delta[i_idx, j_idx] = pt[i_job, bottleneck_machine] + pt[j_job, bottleneck_machine]
        # Normalize
        baseline = np.mean(delta)
        delta -= baseline
    
    # Construct multiple solutions using nearest-neighbor with different starting points
    solutions = []
    for _ in range(num_constructs):
        if len(remaining_jobs) <= 1:
            solutions.append(np.array([first_job] + remaining_jobs))
            continue
        
        # Random start job from remaining
        start_idx = rng.choice(n_rem)
        current = remaining_jobs[start_idx]
        sequence = [first_job, current]
        visited = {start_idx}
        
        for _ in range(n_rem - 1):
            # Find nearest unvisited job
            best_next = -1
            best_cost = float('inf')
            for next_idx, next_job in enumerate(remaining_jobs):
                if next_idx in visited:
                    continue
                cost = delta[remaining_jobs.index(current), next_idx]
                if cost < best_cost:
                    best_cost = cost
                    best_next = next_job
            if best_next == -1:
                break
            sequence.append(best_next)
            visited.add(remaining_jobs.index(best_next))
            current = best_next
        
        # Add any unvisited jobs
        for j in remaining_jobs:
            if j not in sequence:
                sequence.append(j)
        
        solutions.append(np.array(sequence))
    
    # Return the best solution
    best_sol = solutions[0]
    best_ms = _makespan(best_sol, pt)
    for sol in solutions[1:]:
        ms = _makespan(sol, pt)
        if ms < best_ms:
            best_ms = ms
            best_sol = sol
    
    return best_sol

def _insertion_ls(perm, pt, time_limit, start_time):
    """Insertion local search: try moving each job to a better position.
    Uses first-improvement for efficiency."""
    n = len(perm)
    if n <= 1:
        return perm.copy(), _makespan(perm, pt)
    
    best_perm = perm.copy()
    best_ms = _makespan(best_perm, pt)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        # Find the best single move in this pass
        best_move_i = -1
        best_move_j = -1
        best_move_ms = best_ms
        
        for i in range(n):
            if time.time() - start_time >= time_limit:
                break
            job = best_perm[i]
            for j in range(n):
                if j == i:
                    continue
                # Create new permutation
                remaining = np.delete(best_perm, i)
                insert_j = j - 1 if j > i else j
                if insert_j < 0 or insert_j >= len(remaining):
                    continue
                new_perm = np.insert(remaining, insert_j, job)
                ms = _makespan(new_perm, pt)
                if ms < best_move_ms:
                    best_move_ms = ms
                    best_move_i = i
                    best_move_j = j
        
        if best_move_i >= 0:
            # Apply the best move
            job = best_perm[best_move_i]
            remaining = np.delete(best_perm, best_move_i)
            insert_j = best_move_j - 1 if best_move_j > best_move_i else best_move_j
            best_perm = np.insert(remaining, insert_j, job)
            best_ms = best_move_ms
            improved = True
    
    return best_perm, best_ms

def _or_opt_ls(perm, pt, time_limit, start_time):
    """OR-opt local search: move segments of length 1-3 to better positions."""
    n = len(perm)
    if n <= 2:
        return perm.copy(), _makespan(perm, pt)
    
    best_perm = perm.copy()
    best_ms = _makespan(best_perm, pt)
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        best_move = None
        best_move_ms = best_ms
        
        # Try moving segments of length 1, 2, 3
        for seg_len in [1, 2, 3]:
            if time.time() - start_time >= time_limit:
                break
            for i in range(n - seg_len + 1):
                if time.time() - start_time >= time_limit:
                    break
                segment = best_perm[i:i+seg_len].copy()
                # Try inserting at each position
                for j in range(n - seg_len + 1):
                    if j == i:
                        continue
                    # Remove segment
                    remaining = np.concatenate([best_perm[:i], best_perm[i+seg_len:]])
                    # Insert at position j
                    if j > i:
                        j -= seg_len
                    new_perm = np.concatenate([remaining[:j], segment, remaining[j:]])
                    ms = _makespan(new_perm, pt)
                    if ms < best_move_ms:
                        best_move_ms = ms
                        best_move = (i, j, seg_len)
        
        if best_move is not None:
            i, j, seg_len = best_move
            segment = best_perm[i:i+seg_len].copy()
            remaining = np.concatenate([best_perm[:i], best_perm[i+seg_len:]])
            if j > i:
                j -= seg_len
            best_perm = np.concatenate([remaining[:j], segment, remaining[j:]])
            best_ms = best_move_ms
            improved = True
    
    return best_perm, best_ms

def _order_crossover(p1, p2):
    """Order Crossover (OX) for permutations."""
    n = len(p1)
    if n <= 2:
        return p1.copy()
    
    # Pick two random cut points
    i, j = sorted(np.random.choice(n, size=2, replace=False))
    
    # Segment from p1
    segment = p1[i:j+1].copy()
    
    # Fill remaining positions with elements from p2 in order
    remaining = [x for x in p2 if x not in segment]
    
    child = np.empty(n, dtype=int)
    child[i:j+1] = segment
    
    # Fill positions before i and after j
    pos = 0
    for k in range(n):
        if i <= k <= j:
            continue
        child[k] = remaining[pos]
        pos += 1
    
    return child

def _ga_mutation(perm, rng):
    """Swap mutation: randomly swap two jobs."""
    n = len(perm)
    if n <= 1:
        return perm.copy()
    i, j = rng.choice(n, size=2, replace=False)
    new_perm = perm.copy()
    new_perm[i], new_perm[j] = new_perm[j], new_perm[i]
    return new_perm

def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on a flow-shop instance."""
    start_time = time.time()
    deadline = 150.0
    safety_margin = 10.0
    effective_deadline = deadline - safety_margin
    
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
    
    rng = np.random.default_rng(42)
    
    # Generate initial solutions using PFS decomposition and heuristics
    pop_size = min(20, max(10, n // 2))
    population = []
    
    # PFS decomposition constructions (multiple variants)
    for _ in range(3):
        population.append(_pfs_decomposition_construct(processing_times, rng, num_constructs=1))
    
    # Add heuristic solutions
    population.append(_neh_initial(processing_times))
    population.append(_palmer_initial(processing_times))
    population.append(_bottleneck_initial(processing_times))
    
    # Add random solutions
    while len(population) < pop_size:
        population.append(_random_initial(n, rng))
    
    # Evaluate initial population
    fitness = [_makespan(perm, processing_times) for perm in population]
    
    best_perm = population[np.argmin(fitness)].copy()
    best_ms = min(fitness)
    
    # Apply insertion local search to the best initial solution
    best_perm, best_ms = _insertion_ls(best_perm, processing_times, 
                                        time_limit=effective_deadline - (time.time() - start_time),
                                        start_time=start_time)
    
    # OR-opt local search for segment moves
    if time.time() - start_time < effective_deadline:
        remaining = effective_deadline - (time.time() - start_time)
        if remaining > 2:
            best_perm, best_ms = _or_opt_ls(best_perm, processing_times,
                                             time_limit=remaining * 0.5,
                                             start_time=start_time)
    
    # Genetic algorithm with OX crossover
    elapsed = time.time() - start_time
    if elapsed < effective_deadline:
        remaining_time = effective_deadline - elapsed
        # Budget for GA: use most of remaining time
        ga_time_limit = remaining_time * 0.6
        
        # GA parameters
        elitism = 2
        mutation_rate = 0.1
        
        for gen in range(1000):
            if time.time() - start_time >= start_time + ga_time_limit:
                break
            
            # Evaluate current population
            fitness = [_makespan(perm, processing_times) for perm in population]
            
            # Update best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_ms:
                best_ms = fitness[best_idx]
                best_perm = population[best_idx].copy()
            
            # Create next generation
            next_pop = []
            
            # Elitism: keep best individuals
            sorted_indices = np.argsort(fitness)
            for i in range(elitism):
                next_pop.append(population[sorted_indices[i]].copy())
            
            # Fill rest with offspring
            while len(next_pop) < pop_size:
                if time.time() - start_time >= start_time + ga_time_limit:
                    break
                
                # Tournament selection
                def tournament_select(pop, fit, k=3):
                    candidates = rng.choice(len(pop), size=min(k, len(pop)), replace=False)
                    best_cand = candidates[np.argmin([fit[c] for c in candidates])]
                    return pop[best_cand]
                
                p1 = tournament_select(population, fitness)
                p2 = tournament_select(population, fitness)
                
                child = _order_crossover(p1, p2)
                
                # Mutation
                if rng.random() < mutation_rate:
                    child = _ga_mutation(child, rng)
                
                next_pop.append(child)
            
            population = next_pop[:pop_size]
        
        # Final local search on GA result
        if time.time() - start_time < effective_deadline:
            remaining = effective_deadline - (time.time() - start_time)
            if remaining > 2:
                best_perm, best_ms = _insertion_ls(best_perm, processing_times,
                                                    time_limit=remaining * 0.6,
                                                    start_time=start_time)
                if time.time() - start_time < effective_deadline:
                    remaining = effective_deadline - (time.time() - start_time)
                    if remaining > 1:
                        best_perm, best_ms = _or_opt_ls(best_perm, processing_times,
                                                         time_limit=remaining * 0.5,
                                                         start_time=start_time)
    
    # Final polish: one more insertion LS pass if time remains
    if time.time() - start_time < effective_deadline:
        remaining = effective_deadline - (time.time() - start_time)
        if remaining > 1:
            best_perm, best_ms = _insertion_ls(best_perm, processing_times,
                                                time_limit=remaining * 0.9,
                                                start_time=start_time)
    
    return list(best_perm)
# EVOLVE-BLOCK-END
