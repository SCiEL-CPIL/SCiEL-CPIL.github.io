"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan(perm, pt):
    """Compute makespan for a given permutation using the flow-shop recurrence.
    pt: (n, m) array of processing times.
    perm: array of job indices in order.
    Returns makespan (scalar).
    """
    n = len(perm)
    if n == 0:
        return 0
    # C[j][m] = completion time of job j on machine m
    # We only need the last machine's completion times, but we need to track
    # per-machine. Use a 1D array for current machine completions.
    # Actually, we need C[j][m] for all j and m to compute the next.
    # C[j][m] = max(C[j-1][m], C[j][m-1]) + pt[j][m]
    # We can use a 1D array of length m, updated in place.
    # Let comp[m] = C[j][m] for the current job j being processed.
    # But we also need C[j-1][m] which is the previous job's value on machine m.
    # So we need two arrays or update carefully.
    
    # Actually, the standard vectorized approach:
    # For job j in perm:
    #   For machine m in range(m):
    #       C[j][m] = max(C[j-1][m], C[j][m-1]) + pt[perm[j]][m]
    #   where C[-1][m] = 0 and C[j][-1] = 0
    
    # We can do this with a single array 'comp' of length m, where comp[m] holds
    # C[j][m] after processing job j. But we need C[j-1][m] which was the old comp[m].
    # So: new_comp[m] = max(old_comp[m], new_comp[m-1]) + pt[perm[j]][m]
    # We can update in place if we go from m=0 to m=m-1, because new_comp[m-1] is already updated.
    
    m = pt.shape[1]
    comp = np.zeros(m, dtype=float)
    for j in range(n):
        job = perm[j]
        for mi in range(m):
            prev_machine = comp[mi - 1] if mi > 0 else 0.0
            same_machine_prev = comp[mi]  # This is C[j-1][mi]
            comp[mi] = max(same_machine_prev, prev_machine) + pt[job, mi]
    return comp[-1]

def _makespan_vec(perm, pt):
    """Vectorized makespan computation using numpy.
    For small n and m, the loop version might be faster, but let's try a vectorized approach.
    Actually, the recurrence is inherently sequential per job, so full vectorization is hard.
    We'll use the loop version but with numpy operations inside.
    """
    n = len(perm)
    if n == 0:
        return 0.0
    m = pt.shape[1]
    comp = np.zeros(m, dtype=float)
    for j in range(n):
        job = perm[j]
        pt_row = pt[job]
        # new_comp[m] = max(comp[m], new_comp[m-1]) + pt_row[m]
        # We can do this with a loop over m, or use numpy.accumulate with a custom function.
        # For simplicity and speed, loop over m.
        for mi in range(m):
            prev = comp[mi - 1] if mi > 0 else 0.0
            comp[mi] = max(comp[mi], prev) + pt_row[mi]
    return comp[-1]

def order_jobs(processing_times):
    """Construct a job permutation minimizing makespan on a flow-shop instance."""
    start_time = time.time()
    deadline = 150.0  # seconds, with safety margin we'll use 140
    safety_margin = 10.0
    effective_deadline = deadline - safety_margin
    
    n, m = processing_times.shape
    if n <= 1:
        return list(range(n))
    
    # Initialize best solution with NEH heuristic
    total_pt = processing_times.sum(axis=1)
    best_perm = np.argsort(-total_pt).astype(int)
    best_ms = _makespan_vec(best_perm, processing_times)
    
    # Set up random seed for reproducibility within a run
    rng = np.random.default_rng(42)
    
    # Genetic Algorithm with Order Crossover (OX)
    pop_size = min(200, max(50, 2 * n))
    # Initial population: NEH + random permutations + bottleneck-based
    population = [best_perm.copy()]
    
    # Add NEH-based variants
    for _ in range(min(pop_size // 4, 20)):
        # Slight perturbation of NEH
        perm = best_perm.copy()
        for _ in range(min(n // 2, 10)):
            i, j = rng.integers(0, n, 2)
            perm[i], perm[j] = perm[j], perm[i]
        population.append(perm)
    
    # Add random permutations
    while len(population) < pop_size:
        perm = rng.permutation(n)
        population.append(perm)
    
    # Evaluate initial population
    pop_fitness = [_makespan_vec(perm, processing_times) for perm in population]
    best_idx = int(np.argmin(pop_fitness))
    best_perm = population[best_idx].copy()
    best_ms = pop_fitness[best_idx]
    
    # GA parameters
    tournament_size = 5
    mutation_rate = 0.2
    elite_count = max(2, pop_size // 10)
    
    generation = 0
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < effective_deadline:
        # Check deadline
        if time.time() > effective_deadline:
            break
        
        # Create new population
        new_population = []
        
        # Elitism: copy best solutions
        sorted_indices = np.argsort(pop_fitness)
        for i in range(elite_count):
            new_population.append(population[sorted_indices[i]].copy())
        
        # Generate offspring via OX crossover
        while len(new_population) < pop_size:
            # Tournament selection
            def tournament_select():
                indices = rng.integers(0, pop_size, tournament_size)
                fits = [pop_fitness[i] for i in indices]
                best_i = indices[int(np.argmin(fits))]
                return population[best_i].copy()
            
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            # Order Crossover (OX1)
            if n < 2:
                child = parent1.copy()
            else:
                c1, c2 = rng.integers(0, n, 2)
                if c1 > c2:
                    c1, c2 = c2, c1
                child = np.full(n, -1, dtype=int)
                child[c1:c2+1] = parent1[c1:c2+1]
                # Fill remaining positions with elements from parent2 in order
                remaining = [j for j in range(n) if j not in parent1[c1:c2+1]]
                pos = (c2 + 1) % n
                for gene in parent2:
                    if child[pos] == -1:
                        if gene in remaining:
                            remaining.remove(gene)
                        child[pos] = gene
                        pos = (pos + 1) % n
                # If there are still -1s (shouldn't happen), fill with remaining
                for i in range(n):
                    if child[i] == -1:
                        if remaining:
                            child[i] = remaining.pop(0)
            
            # Mutation: swap mutation and insertion mutation
            if rng.random() < mutation_rate:
                if n >= 2:
                    i, j = rng.integers(0, n, 2)
                    child[i], child[j] = child[j], child[i]
            if rng.random() < mutation_rate:
                if n >= 2:
                    i, j = rng.integers(0, n, 2)
                    if i != j:
                        gene = child[i]
                        if i < j:
                            child[i:j] = child[i+1:j+1]
                            child[j] = gene
                        else:
                            child[j+1:i+1] = child[j:i]
                            child[i] = gene
            
            new_population.append(child)
        
        # Evaluate new population
        pop_fitness = [_makespan_vec(perm, processing_times) for perm in new_population]
        population = new_population
        
        # Update best
        gen_best_idx = int(np.argmin(pop_fitness))
        if pop_fitness[gen_best_idx] < best_ms:
            best_ms = pop_fitness[gen_best_idx]
            best_perm = population[gen_best_idx].copy()
            no_improve = 0
        else:
            no_improve += 1
        
        generation += 1
        
        # If no improvement for a while, add some random restarts
        if no_improve >= max_no_improve:
            for _ in range(min(5, pop_size // 10)):
                population[rng.integers(0, pop_size)] = rng.permutation(n)
            no_improve = 0
    
    return list(best_perm)
# EVOLVE-BLOCK-END
