# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(perm, pt):
    """Compute makespan for a given permutation. O(n*m) using the flow-shop recurrence."""
    n = len(perm)
    if n == 0:
        return 0
    m = pt.shape[1]
    # C[i][j] = completion time of job i on machine j
    # C[i][0] = C[i-1][0] + pt[perm[i]][0]
    # C[i][j] = max(C[i-1][j], C[i][j-1]) + pt[perm[i]][j]
    # We can do this in O(n*m) with a 1D array
    C = np.zeros(m)
    for i in range(n):
        job = perm[i]
        for j in range(m):
            if i == 0:
                if j == 0:
                    C[j] = pt[job, j]
                else:
                    C[j] = C[j-1] + pt[job, j]
            else:
                if j == 0:
                    C[j] = C[j] + pt[job, j]
                else:
                    C[j] = max(C[j], C[j-1]) + pt[job, j]
    return C[-1]


def _makespan_vec(perm, pt):
    """Vectorized makespan computation. Still O(n*m) but with less Python overhead."""
    n = len(perm)
    m = pt.shape[1]
    if n == 0:
        return 0
    if m == 0:
        return 0
    # C[i][j] recurrence
    # Use a single row, updating in place
    C = np.zeros(m)
    for i in range(n):
        job = perm[i]
        row = pt[job]
        if i == 0:
            C[0] = row[0]
            if m > 1:
                C[1:] = np.cumsum(row[1:])
        else:
            C[0] += row[0]
            if m > 1:
                # C[j] = max(C[j], C[j-1]) + row[j]
                # This is a sequential dependency, but we can do it in a loop
                for j in range(1, m):
                    C[j] = max(C[j], C[j-1]) + row[j]
    return C[-1]


def _neh_order(pt):
    """NEH heuristic for initial solution."""
    n = pt.shape[0]
    m = pt.shape[1]
    totals = pt.sum(axis=1)
    # Sort jobs by total processing time descending
    order = np.argsort(-totals)
    
    # Build the sequence using NEH insertion
    seq = [order[0]]
    for k in range(1, n):
        job = order[k]
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(seq) + 1):
            candidate = seq[:pos] + [job] + seq[pos:]
            ms = _makespan(candidate, pt)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        seq = seq[:best_pos] + [job] + seq[best_pos:]
    return seq


def _ox_crossover(p1, p2):
    """Order Crossover (OX)."""
    n = len(p1)
    if n <= 2:
        return list(p1), list(p2)
    
    a, b = sorted(random.sample(range(n), 2))
    
    child1 = [None] * n
    child2 = [None] * n
    
    # Copy segment from p1 to child1
    child1[a:b+1] = p1[a:b+1]
    # Copy segment from p2 to child2
    child2[a:b+1] = p2[a:b+1]
    
    # Fill rest from p2, maintaining order
    fill1 = [x for x in p2 if x not in child1[a:b+1]]
    pos = 0
    for i in range(n):
        if i < a or i > b:
            child1[i] = fill1[pos]
            pos += 1
    
    # Fill rest from p1, maintaining order
    fill2 = [x for x in p1 if x not in child2[a:b+1]]
    pos = 0
    for i in range(n):
        if i < a or i > b:
            child2[i] = fill2[pos]
            pos += 1
    
    return child1, child2


def _swap_mutation(perm, rate=0.1):
    """Swap mutation: swap two random positions."""
    n = len(perm)
    if n <= 1:
        return list(perm)
    child = list(perm)
    if random.random() < rate:
        i, j = random.sample(range(n), 2)
        child[i], child[j] = child[j], child[i]
    return child


def _insertion_mutation(perm, rate=0.1):
    """Insertion mutation: move a random job to a random position."""
    n = len(perm)
    if n <= 1:
        return list(perm)
    child = list(perm)
    if random.random() < rate:
        i = random.randint(0, n-1)
        job = child.pop(i)
        j = random.randint(0, len(child))
        child.insert(j, job)
    return child


def _local_search(perm, pt, max_iter=50, time_limit=None):
    """Simple local search with 2-opt moves."""
    n = len(perm)
    if n <= 1:
        return list(perm)
    
    current = list(perm)
    current_ms = _makespan(current, pt)
    best = list(current)
    best_ms = current_ms
    
    start = time.time()
    iters = 0
    while iters < max_iter:
        if time_limit is not None and time.time() - start > time_limit:
            break
        improved = False
        for i in range(n):
            for j in range(i+1, n):
                candidate = list(current)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                ms = _makespan(candidate, pt)
                if ms < current_ms - 1e-10:
                    current = candidate
                    current_ms = ms
                    improved = True
                    if current_ms < best_ms - 1e-10:
                        best = list(current)
                        best_ms = current_ms
        if not improved:
            break
        iters += 1
    
    return best


def order_jobs(processing_times):
    """
    Genetic Algorithm with Order Crossover for flow-shop scheduling.
    """
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    pt = np.asarray(processing_times, dtype=np.float64)
    n = pt.shape[0]
    m = pt.shape[1]
    
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    # Initial population: NEH + random permutations
    pop_size = min(100, max(20, n * 5))
    
    # First solution: NEH
    neh_sol = _neh_order(pt)
    neh_ms = _makespan(neh_sol, pt)
    
    # Build initial population
    population = [neh_sol]
    fitnesses = [neh_ms]
    
    # Add random permutations
    for _ in range(pop_size - 1):
        perm = list(range(n))
        random.shuffle(perm)
        population.append(perm)
        fitnesses.append(_makespan(perm, pt))
    
    # Track best
    best_idx = int(np.argmin(fitnesses))
    best_sol = list(population[best_idx])
    best_ms = fitnesses[best_idx]
    
    # GA loop
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Selection: tournament
        def tournament_select(k=3):
            idxs = random.sample(range(len(population)), k)
            return min(idxs, key=lambda i: fitnesses[i])
        
        # Create new population
        new_population = []
        new_fitnesses = []
        
        # Elitism: keep top 10%
        elite_count = max(1, len(population) // 10)
        sorted_indices = np.argsort(fitnesses)
        for i in sorted_indices[:elite_count]:
            new_population.append(list(population[i]))
            new_fitnesses.append(fitnesses[i])
        
        # Fill rest with GA offspring
        while len(new_population) < pop_size:
            p1 = tournament_select()
            p2 = tournament_select()
            
            # Crossover
            if random.random() < 0.8:
                c1, c2 = _ox_crossover(population[p1], population[p2])
            else:
                c1 = list(population[p1])
                c2 = list(population[p2])
            
            # Mutation
            c1 = _swap_mutation(c1, rate=0.15)
            c1 = _insertion_mutation(c1, rate=0.15)
            c2 = _swap_mutation(c2, rate=0.15)
            c2 = _insertion_mutation(c2, rate=0.15)
            
            new_population.append(c1)
            new_fitnesses.append(_makespan(c1, pt))
            
            if len(new_population) < pop_size:
                new_population.append(c2)
                new_fitnesses.append(_makespan(c2, pt))
        
        population = new_population
        fitnesses = new_fitnesses
        
        # Update best
        idx = int(np.argmin(fitnesses))
        if fitnesses[idx] < best_ms - 1e-10:
            best_ms = fitnesses[idx]
            best_sol = list(population[idx])
        
        # Periodic local search on the best solution
        if generation % 5 == 0 and time.time() < deadline - 5.0:
            ls_time_limit = min(5.0, deadline - time.time() - 1.0)
            if ls_time_limit > 0.5:
                ls_result = _local_search(best_sol, pt, max_iter=30, time_limit=ls_time_limit)
                ls_ms = _makespan(ls_result, pt)
                if ls_ms < best_ms - 1e-10:
                    best_ms = ls_ms
                    best_sol = ls_result
    
    return best_sol


# EVOLVE-BLOCK-END
