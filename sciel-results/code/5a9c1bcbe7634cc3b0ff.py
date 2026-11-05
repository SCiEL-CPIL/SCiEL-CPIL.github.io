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

def _insertion_ls(perm, pt, time_limit, start_time):
    """Insertion local search: try moving each job to a better position.
    Uses best-improvement within each pass for efficiency."""
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
    
    # Generate initial population using diverse heuristics
    pop_size = min(20, max(10, n // 2))
    population = []
    
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
    
    # Genetic algorithm with OX crossover
    elapsed = time.time() - start_time
    if elapsed < effective_deadline:
        remaining_time = effective_deadline - elapsed
        # Budget for GA: use most of remaining time
        ga_time_limit = remaining_time * 0.7
        
        # GA parameters
        gen_size = min(15, pop_size)
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
                                                    time_limit=remaining * 0.8,
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
