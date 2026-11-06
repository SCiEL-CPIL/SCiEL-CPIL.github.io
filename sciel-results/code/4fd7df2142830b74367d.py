"""
Strategy:
Instead of pruning a fixed Batcher network, use a constructive genetic algorithm (GA) that
evolves the network topology from scratch. The fitness function is based on the reduction of
unsorted 0-1 patterns (using the Zero-One Principle). This explores a broader search space
of topologies that may not be reachable via simple deletion.

Key components:
1. Representation: List of comparator pairs (i, j).
2. Initialization: A mix of random networks and a known good structure (e.g., Batcher or odd-even).
3. Fitness:
   - For a given network, apply it to all 2^16 boolean inputs.
   - Count the number of inputs that are NOT sorted.
   - Fitness = -count (or count, if minimizing).
   - Valid networks have 0 unsorted inputs.
4. Selection: Tournament selection based on fitness.
5. Crossover: Uniform crossover or single-point crossover on the list of comparators.
   - Note: Crossover may produce invalid networks (e.g., duplicate comparators, out-of-order).
   - We treat the list as a multiset or sequence.
6. Mutation:
   - Insert a random comparator.
   - Remove a random comparator.
   - Replace a random comparator.
   - Swap two comparators.
7. Elitism: Keep the best individuals.
8. Deadline guard: Return the best valid network found.

To make this efficient:
- Use vectorized numpy operations for fitness evaluation.
- Pre-compute the boolean inputs.
- Limit population size and generations based on time.
"""

import time
import numpy as np

N = 16
MAX_TIME = 58.0

def _generate_batcher_network():
    """
    Generates Batcher's Odd-Even Merge Sort network for N=16.
    Returns a flat list of comparators.
    """
    def merge(n, offset):
        if n == 1:
            return [(offset, offset + 1)]
        
        h = n // 2
        step1 = [(offset + i, offset + i + n) for i in range(n)]
        
        def strided_merge(n, offset, stride):
            if n == 1:
                return [(offset, offset + stride)]
            
            h = n // 2
            step1 = [(offset + i * stride, offset + (n + i) * stride) for i in range(n)]
            even = strided_merge(h, offset, 2 * stride)
            odd = strided_merge(h, offset + stride, 2 * stride)
            return step1 + even + odd
        
        even_merge = strided_merge(h, offset, 2)
        odd_merge = strided_merge(h, offset + 1, 2)
        
        return step1 + even_merge + odd_merge

    def sort(n, offset):
        if n <= 1:
            return []
        h = n // 2
        left = sort(h, offset)
        right = sort(h, offset + h)
        m = merge(h, offset)
        return left + right + m

    return sort(N, 0)

def _apply_network_vectorized(comps, inputs):
    """
    Apply the network to a batch of inputs.
    """
    vec = inputs.copy()
    for i, j in comps:
        min_val = np.minimum(vec[:, i], vec[:, j])
        max_val = np.maximum(vec[:, i], vec[:, j])
        vec[:, i] = min_val
        vec[:, j] = max_val
    return vec

def _count_unsorted(comps, bits):
    """
    Count the number of unsorted boolean inputs.
    Returns 0 if the network is valid.
    """
    out = _apply_network_vectorized(comps, bits)
    diffs = np.diff(out, axis=1)
    # A row is unsorted if any diff is negative
    unsorted_rows = np.any(diffs < 0, axis=1)
    return np.sum(unsorted_rows)

def construct():
    start_time = time.time()
    deadline = start_time + MAX_TIME
    
    # Pre-compute all boolean inputs
    n_inputs = 1 << N
    indices = np.arange(n_inputs)
    bits = (indices[:, None] >> np.arange(N)[::-1]) & 1
    
    # All possible comparators
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    n_all_comps = len(all_comps)
    
    rng = np.random.default_rng(42)
    
    # Population parameters
    pop_size = 100
    elite_size = 5
    
    # Initialize population
    population = []
    
    # Add a known good network (Batcher)
    batcher = _generate_batcher_network()
    population.append(batcher)
    
    # Add some random networks
    for _ in range(pop_size - 1):
        # Random length between 40 and 80
        length = rng.integers(40, 80)
        comp_list = []
        for _ in range(length):
            comp = all_comps[rng.integers(0, n_all_comps)]
            comp_list.append(comp)
        population.append(comp_list)
    
    # Evaluate initial population
    fitnesses = []
    for ind in population:
        unsorted = _count_unsorted(ind, bits)
        fitnesses.append(unsorted)
    
    # Sort by fitness (ascending, since we want to minimize unsorted count)
    sorted_indices = np.argsort(fitnesses)
    population = [population[i] for i in sorted_indices]
    fitnesses = [fitnesses[i] for i in sorted_indices]
    
    best_network = population[0]
    best_fitness = fitnesses[0]
    best_len = len(best_network)
    
    # GA Loop
    while time.time() < deadline:
        if time.time() > deadline - 1.0:
            break
            
        new_population = []
        
        # Elitism
        for i in range(elite_size):
            new_population.append(population[i][:])
        
        # Generate offspring
        while len(new_population) < pop_size:
            # Tournament selection
            idx1 = rng.integers(0, pop_size)
            idx2 = rng.integers(0, pop_size)
            if fitnesses[idx1] < fitnesses[idx2]:
                parent1 = population[idx1]
            else:
                parent1 = population[idx2]
                
            idx1 = rng.integers(0, pop_size)
            idx2 = rng.integers(0, pop_size)
            if fitnesses[idx1] < fitnesses[idx2]:
                parent2 = population[idx1]
            else:
                parent2 = population[idx2]
            
            # Crossover
            # Simple: take a random prefix from parent1 and random suffix from parent2
            # But lengths might differ. Let's pad or truncate.
            # Or just random mixing.
            child = []
            len1 = len(parent1)
            len2 = len(parent2)
            
            # Random length for child
            child_len = rng.integers(min(len1, len2), max(len1, len2) + 10)
            
            # Mix comparators
            for _ in range(child_len):
                if rng.random() < 0.5:
                    if len1 > 0:
                        comp = parent1[rng.integers(0, len1)]
                        child.append(comp)
                else:
                    if len2 > 0:
                        comp = parent2[rng.integers(0, len2)]
                        child.append(comp)
            
            # Mutation
            # 1. Remove random comparator
            if rng.random() < 0.3 and len(child) > 0:
                idx = rng.integers(0, len(child))
                child.pop(idx)
            
            # 2. Insert random comparator
            if rng.random() < 0.3:
                comp = all_comps[rng.integers(0, n_all_comps)]
                idx = rng.integers(0, len(child) + 1)
                child.insert(idx, comp)
            
            # 3. Replace random comparator
            if rng.random() < 0.3 and len(child) > 0:
                idx = rng.integers(0, len(child))
                child[idx] = all_comps[rng.integers(0, n_all_comps)]
            
            new_population.append(child)
        
        # Evaluate new population
        new_fitnesses = []
        for ind in new_population:
            unsorted = _count_unsorted(ind, bits)
            new_fitnesses.append(unsorted)
        
        # Sort by fitness
        sorted_indices = np.argsort(new_fitnesses)
        population = [new_population[i] for i in sorted_indices]
        fitnesses = [new_fitnesses[i] for i in sorted_indices]
        
        # Update best
        if fitnesses[0] < best_fitness:
            best_fitness = fitnesses[0]
            best_network = population[0]
            best_len = len(best_network)
        
        # If we found a valid network, try to optimize its length
        if best_fitness == 0:
            # Try to reduce the length of the best network
            # This is a local search on the best individual
            improved = False
            for _ in range(10): # Try a few times
                if time.time() > deadline - 1.0:
                    break
                current = best_network[:]
                idx = rng.integers(0, len(current))
                current.pop(idx)
                unsorted = _count_unsorted(current, bits)
                if unsorted == 0:
                    best_network = current
                    best_len = len(best_network)
                    improved = True
                else:
                    # Try replacement
                    new_comp = all_comps[rng.integers(0, n_all_comps)]
                    current[idx] = new_comp
                    unsorted = _count_unsorted(current, bits)
                    if unsorted == 0:
                        best_network = current
                        best_len = len(best_network)
                        improved = True
            
            if not improved:
                # If no improvement, continue GA but focus on valid networks
                pass
    
    return best_network

# EVOLVE-BLOCK-END
