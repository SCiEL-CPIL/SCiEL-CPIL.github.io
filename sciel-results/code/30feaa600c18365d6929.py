"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 256
DEADLINE_MARGIN = 15.0  # safety margin before the 150s kill

def _count_mono4_fast(colours):
    """
    Faster counting using numpy. colours is a numpy array of 0/1 of length 256.
    """
    total = 0
    n = 256
    for d in range(1, n // 3 + 1):
        max_a = n - 3 * d
        c0 = colours[0:max_a]
        c1 = colours[d:d + max_a]
        c2 = colours[2 * d:2 * d + max_a]
        c3 = colours[3 * d:3 * d + max_a]
        mask = (c0 == c1) & (c0 == c2) & (c0 == c3)
        total += np.count_nonzero(mask)
    return int(total)

def _thue_morse(n):
    """Generate Thue-Morse sequence of length n using popcount."""
    c = np.zeros(n, dtype=np.int8)
    for i in range(n):
        c[i] = bin(i).count('1') % 2
    return c

def _precompute_ap_membership():
    """Precompute which APs contain each position.
    Returns list of lists: ap_list[i] = list of (a, d) tuples where position i is in AP (a, a+d, a+2d, a+3d)."""
    ap_list = [[] for _ in range(N)]
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        if max_a <= 0:
            break
        for a in range(max_a):
            for k in range(4):
                pos = a + k * d
                ap_list[pos].append((a, d))
    return ap_list

def _delta_flip(c, i, ap_list):
    """Compute change in mono-4AP count if we flip position i, using precomputed membership."""
    delta = 0
    for (a, d) in ap_list[i]:
        idxs = [a, a + d, a + 2 * d, a + 3 * d]
        vals = [c[j] for j in idxs]
        was_mono = (vals[0] == vals[1] == vals[2] == vals[3])
        # After flip: position i has flipped
        new_vals = [c[j] if j != i else 1 - c[j] for j in idxs]
        is_mono = (new_vals[0] == new_vals[1] == new_vals[2] == new_vals[3])
        if was_mono and not is_mono:
            delta -= 1
        elif not was_mono and is_mono:
            delta += 1
    return delta

def _local_search(c, time_limit, ap_list):
    """Improve coloring via single-flip local search using delta computation."""
    start = time.time()
    current = _count_mono4_fast(c)
    improved = True
    while improved:
        improved = False
        for i in range(N):
            if time.time() - start > time_limit:
                break
            delta = _delta_flip(c, i, ap_list)
            if delta < 0:
                c[i] = 1 - c[i]
                current += delta
                improved = True
        if time.time() - start > time_limit:
            break
    return c, current

def _swap_search(c, time_limit, ap_list):
    """Try swapping pairs of positions with different colors."""
    start = time.time()
    best_c = c.copy()
    best_count = _count_mono4_fast(best_c)
    
    improved = True
    while improved and (time.time() - start) < time_limit:
        improved = False
        for i in range(N):
            if (time.time() - start) >= time_limit:
                break
            for j in range(i + 1, N):
                if (time.time() - start) >= time_limit:
                    break
                if best_c[i] != best_c[j]:
                    # Swapping would change both positions
                    best_c[i], best_c[j] = best_c[j], best_c[i]
                    new_count = _count_mono4_fast(best_c)
                    if new_count < best_count:
                        best_count = new_count
                        improved = True
                    else:
                        best_c[i], best_c[j] = best_c[j], best_c[i]
    
    return best_c, best_count

def _crossover(p1, p2, rng):
    """Uniform crossover: for each position, pick from p1 or p2 with 50% probability."""
    mask = rng.integers(0, 2, size=N).astype(np.int8)
    child = np.where(mask == 1, p1, p2)
    return child.copy()

def _mutate(child, rng, num_mutations=5):
    """Randomly flip some bits."""
    if num_mutations > 0:
        mut_positions = rng.choice(N, size=num_mutations, replace=False)
        for mp in mut_positions:
            child[mp] = 1 - child[mp]
    return child

def construct():
    start_time = time.time()
    deadline = start_time + 150.0 - DEADLINE_MARGIN
    
    # Precompute AP membership for fast delta computation
    ap_list = _precompute_ap_membership()
    
    # Generate diverse initial solutions
    initial_solutions = []
    
    # Thue-Morse and variants
    tm = _thue_morse(N)
    initial_solutions.append(tm.copy())
    
    # Thue-Morse with offsets
    for offset in range(0, 8):
        tm_offset = np.roll(tm, offset)
        initial_solutions.append(tm_offset.copy())
    
    # Alternating patterns
    for block_size in [1, 2, 3, 4, 5, 6, 7, 8, 16, 32]:
        pattern = np.array([(i // block_size) % 2 for i in range(N)], dtype=np.int8)
        initial_solutions.append(pattern.copy())
    
    # Random solutions
    rng = np.random.default_rng(42)
    for seed in range(10):
        rand_sol = rng.integers(0, 2, size=N).astype(np.int8)
        initial_solutions.append(rand_sol)
    
    # Evaluate all initial solutions
    scored_solutions = []
    for sol in initial_solutions:
        score = _count_mono4_fast(sol)
        scored_solutions.append((score, sol.copy()))
    
    # Sort by score
    scored_solutions.sort(key=lambda x: x[0])
    
    best_score = scored_solutions[0][0]
    best_colours = scored_solutions[0][1].copy()
    
    # Refine top candidates with local search
    top_k = min(5, len(scored_solutions))
    for i in range(top_k):
        if time.time() > deadline - 30.0:
            break
        remaining = min(15.0, deadline - time.time() - 5.0)
        if remaining > 2.0:
            refined, refined_score = _local_search(scored_solutions[i][1].copy(), remaining, ap_list)
            if refined_score < best_score:
                best_score = refined_score
                best_colours = refined.copy()
    
    # Genetic algorithm
    population_size = 25
    # Initialize population
    population = []
    for i in range(population_size):
        if i < top_k:
            pop = scored_solutions[i][1].copy()
        else:
            pop = best_colours.copy()
            num_flips = rng.integers(1, 8)
            flip_positions = rng.choice(N, size=num_flips, replace=False)
            for fp in flip_positions:
                pop[fp] = 1 - pop[fp]
        pop_score = _count_mono4_fast(pop)
        population.append((pop_score, pop.copy()))
    
    # Sort population
    population.sort(key=lambda x: x[0])
    
    # GA loop
    generation = 0
    while time.time() < deadline - 20.0:
        generation += 1
        
        # Selection: keep top half
        half_size = population_size // 2
        parents = [pop[1] for pop in population[:half_size]]
        
        # Crossover and mutation to create new offspring
        offspring = []
        for i in range(population_size - half_size):
            # Pick two parents (tournament selection)
            idx1 = rng.integers(0, half_size)
            idx2 = rng.integers(0, half_size)
            p1 = parents[idx1]
            p2 = parents[idx2]
            
            # Uniform crossover
            child = _crossover(p1, p2, rng)
            
            # Mutation
            num_mutations = rng.integers(0, 6)
            child = _mutate(child, rng, num_mutations)
            
            child_score = _count_mono4_fast(child)
            offspring.append((child_score, child.copy()))
        
        # Combine parents and offspring
        population = population[:half_size] + offspring
        population.sort(key=lambda x: x[0])
        
        # Update best
        if population[0][0] < best_score:
            best_score = population[0][0]
            best_colours = population[0][1].copy()
        
        # Local search on the best solution in population
        if time.time() < deadline - 15.0:
            remaining = min(10.0, deadline - time.time() - 10.0)
            if remaining > 2.0:
                refined, refined_score = _local_search(population[0][1].copy(), remaining, ap_list)
                if refined_score < best_score:
                    best_score = refined_score
                    best_colours = refined.copy()
                    population[0] = (refined_score, refined.copy())
                    population.sort(key=lambda x: x[0])
        
        # Occasionally try swap search on the best
        if generation % 5 == 0 and time.time() < deadline - 25.0:
            remaining = min(8.0, deadline - time.time() - 15.0)
            if remaining > 2.0:
                swapped, swap_score = _swap_search(best_colours.copy(), remaining, ap_list)
                if swap_score < best_score:
                    best_score = swap_score
                    best_colours = swapped.copy()
    
    # Final local search polish
    if time.time() < deadline - 5.0:
        remaining = min(20.0, deadline - time.time() - 2.0)
        if remaining > 2.0:
            refined, refined_score = _local_search(best_colours.copy(), remaining, ap_list)
            if refined_score < best_score:
                best_score = refined_score
                best_colours = refined.copy()
    
    # Convert to list of ints
    result = [int(x) for x in best_colours]
    assert len(result) == 256
    assert all(x in (0, 1) for x in result)
    return result
