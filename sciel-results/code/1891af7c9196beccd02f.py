"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random


def _makespan(processing_times, perm):
    """Compute makespan for a given permutation. O(n*m) with minimal overhead."""
    n = len(perm)
    m = processing_times.shape[1]
    # completion[j][i] = completion time of job j on machine i
    # We track only the last machine's completion for each job prefix
    # Actually, for flow shop: C(j, k) = max(C(j, k-1), C(j-1, k)) + P(j, k)
    # We can compute this row by row for the permutation.
    
    # Let's use a 1D array for the current row of completions
    # comp[k] = completion time of the current job on machine k
    # prev_comp[k] = completion time of the previous job on machine k
    
    prev_comp = np.zeros(m)
    for j in range(n):
        job_idx = perm[j]
        p = processing_times[job_idx]
        # For the current job, compute completion on each machine
        # comp[0] = p[0] (since prev_comp[0] is completion of prev job on machine 0, but for first job it's 0)
        # Actually: C(j, 0) = C(j-1, 0) + P(j, 0) for j > 0, and P(0, 0) for j=0
        # C(j, k) = max(C(j, k-1), C(j-1, k)) + P(j, k)
        
        # We need to be careful: comp[k-1] is C(j, k-1), prev_comp[k] is C(j-1, k)
        # So comp[k] = max(comp[k-1], prev_comp[k]) + p[k]
        
        comp = np.empty(m)
        comp[0] = prev_comp[0] + p[0]
        if m > 1:
            # For k >= 1: comp[k] = max(comp[k-1], prev_comp[k]) + p[k]
            # We can do this vectorized
            comp[1:] = np.maximum(comp[:-1], prev_comp[1:]) + p[1:]
        prev_comp = comp
    
    return prev_comp[m-1]


def _neh_order(processing_times):
    """NEH heuristic: order jobs by decreasing total processing time, insert each job in best position."""
    n = processing_times.shape[0]
    m = processing_times.shape[1]
    
    # Sort jobs by decreasing total processing time
    totals = processing_times.sum(axis=1)
    order = np.argsort(-totals)
    
    # Build permutation incrementally
    perm = [order[0]]
    for idx in range(1, n):
        job = order[idx]
        best_pos = 0
        best_ms = float('inf')
        # Try inserting at each position
        for pos in range(len(perm) + 1):
            candidate = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(processing_times, candidate)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm = perm[:best_pos] + [job] + perm[best_pos:]
    
    return perm


def _ox_crossover(p1, p2):
    """Order Crossover (OX)."""
    n = len(p1)
    if n <= 2:
        return p1[:], p2[:]
    
    a, b = sorted(random.sample(range(n), 2))
    
    # Create child1
    child1 = [-1] * n
    child1[a:b+1] = p1[a:b+1]
    # Fill remaining with elements from p2 in order, skipping those already in child1
    in_child1 = set(p1[a:b+1])
    pos = 0
    for i in range(n):
        if p2[i] not in in_child1:
            # Find next empty slot
            while pos < n and child1[pos] != -1:
                pos += 1
            if pos < n:
                child1[pos] = p2[i]
                pos += 1
    
    # Create child2
    child2 = [-1] * n
    child2[a:b+1] = p2[a:b+1]
    in_child2 = set(p2[a:b+1])
    pos = 0
    for i in range(n):
        if p1[i] not in in_child2:
            while pos < n and child2[pos] != -1:
                pos += 1
            if pos < n:
                child2[pos] = p1[i]
                pos += 1
    
    return child1, child2


def _swap_mutation(perm, rate=0.1):
    """Swap mutation."""
    n = len(perm)
    child = perm[:]
    num_swaps = max(1, int(n * rate))
    for _ in range(num_swaps):
        i, j = random.sample(range(n), 2)
        child[i], child[j] = child[j], child[i]
    return child


def _inversion_mutation(perm):
    """Inversion mutation: reverse a random substring."""
    n = len(perm)
    if n <= 1:
        return perm[:]
    i, j = sorted(random.sample(range(n), 2))
    child = perm[:]
    child[i:j+1] = child[i:j+1][::-1]
    return child


def order_jobs(processing_times):
    """
    Genetic Algorithm with Order Crossover for flow-shop scheduling.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    processing_times = np.asarray(processing_times, dtype=np.float64)
    n = processing_times.shape[0]
    m = processing_times.shape[1]
    
    if n == 0:
        return []
    if n == 1:
        return [0]
    
    # Generate initial population
    pop_size = min(100, max(20, n * 2))
    
    # Use NEH for one individual
    neh_perm = _neh_order(processing_times)
    
    # Add random permutations
    population = [neh_perm]
    for _ in range(pop_size - 1):
        perm = list(range(n))
        random.shuffle(perm)
        population.append(perm)
    
    # Evaluate fitness
    fitness = [_makespan(processing_times, perm) for perm in population]
    
    best_perm = population[0]
    best_fitness = fitness[0]
    for i in range(1, pop_size):
        if fitness[i] < best_fitness:
            best_fitness = fitness[i]
            best_perm = population[i]
    
    # GA loop
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Sort by fitness
        indexed = list(range(pop_size))
        indexed.sort(key=lambda i: fitness[i])
        sorted_population = [population[i] for i in indexed]
        sorted_fitness = [fitness[i] for i in indexed]
        
        # Elitism: keep top 2
        new_population = [sorted_population[0][:], sorted_population[1][:]]
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
            
            # Tournament selection
            def tournament_select():
                a, b = random.sample(range(pop_size), 2)
                if fitness[a] < fitness[b]:
                    return population[a]
                else:
                    return population[b]
            
            p1 = tournament_select()
            p2 = tournament_select()
            
            # Crossover
            if random.random() < 0.8:
                c1, c2 = _ox_crossover(p1, p2)
            else:
                c1, c2 = p1[:], p2[:]
            
            # Mutation
            if random.random() < 0.5:
                c1 = _swap_mutation(c1)
            if random.random() < 0.5:
                c2 = _swap_mutation(c2)
            
            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)
        
        # Evaluate new population
        new_fitness = [_makespan(processing_times, perm) for perm in new_population]
        
        # Update best
        for i in range(len(new_fitness)):
            if new_fitness[i] < best_fitness:
                best_fitness = new_fitness[i]
                best_perm = new_population[i]
        
        population = new_population
        fitness = new_fitness
        
        # Check deadline
        if time.time() > deadline:
            break
    
    return best_perm


# EVOLVE-BLOCK-END
