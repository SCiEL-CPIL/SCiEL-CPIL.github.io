"""Genetic algorithm with vectorized scoring and local search polish."""
import numpy as np
import time

def construct():
    N = 256
    deadline = time.time() + 115  # safety margin before 120s limit
    
    # Precompute all 4-term arithmetic progressions as index tuples (0-based)
    aps = []
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Convert to numpy array for fast vectorized operations
    aps_arr = np.array(aps, dtype=np.int32)
    n_aps = len(aps_arr)
    
    # Split into 4 separate index arrays for vectorized access
    i1 = aps_arr[:, 0]
    i2 = aps_arr[:, 1]
    i3 = aps_arr[:, 2]
    i4 = aps_arr[:, 3]
    
    # Precompute which APs contain each index for local search
    aps_containing = [[] for _ in range(N)]
    for idx in range(n_aps):
        for k in range(4):
            aps_containing[aps_arr[idx, k]].append(idx)
    
    def count_mono_4ap_vectorized(population):
        """
        Count monochromatic 4-APs for a population of colorings.
        population: shape (P, N)
        returns: array of shape (P,)
        """
        # population is (P, N)
        # i1, i2, i3, i4 are (A,)
        # We want to compare colors at these indices for all P individuals.
        # c1: (P, A), c2: (P, A), etc.
        c1 = population[:, i1]
        c2 = population[:, i2]
        c3 = population[:, i3]
        c4 = population[:, i4]
        
        # Monochromatic means all four are equal
        # (c1 == c2) & (c2 == c3) & (c3 == c4)
        mono = (c1 == c2) & (c2 == c3) & (c3 == c4)
        return np.sum(mono, axis=1)

    def local_search(colors, max_passes=50):
        """
        Improve a single coloring by flipping bits that reduce the count.
        """
        current_score = count_mono_4ap_vectorized(colors.reshape(1, -1))[0]
        
        # Track which APs are currently monochromatic to speed up delta calculation?
        # Or just re-evaluate the affected APs.
        # Let's re-evaluate affected APs.
        
        improved = True
        passes = 0
        while improved and passes < max_passes and time.time() < deadline - 5:
            improved = False
            passes += 1
            # Random order to break symmetry
            order = np.random.permutation(N)
            for i in order:
                # Try flipping bit i
                old_bit = colors[i]
                new_bit = 1 - old_bit
                
                # Calculate delta
                delta = 0
                for ap_idx in aps_containing[i]:
                    # Get the 4 indices of this AP
                    idxs = aps_arr[ap_idx]
                    # Get current colors
                    c0 = colors[idxs[0]]
                    c1 = colors[idxs[1]]
                    c2 = colors[idxs[2]]
                    c3 = colors[idxs[3]]
                    
                    # Was it mono?
                    was_mono = (c0 == c1) and (c1 == c2) and (c2 == c3)
                    
                    # What will it be?
                    # Only c_i changes
                    new_c0 = new_bit if idxs[0] == i else c0
                    new_c1 = new_bit if idxs[1] == i else c1
                    new_c2 = new_bit if idxs[2] == i else c2
                    new_c3 = new_bit if idxs[3] == i else c3
                    
                    is_mono = (new_c0 == new_c1) and (new_c1 == new_c2) and (new_c2 == new_c3)
                    
                    if was_mono and not is_mono:
                        delta -= 1
                    elif not was_mono and is_mono:
                        delta += 1
                
                if delta < 0:
                    colors[i] = new_bit
                    current_score += delta
                    improved = True
        
        return colors, current_score

    # Initialize population with diverse starting points
    np.random.seed(42)
    pop_size = 100 # Reduced to allow more local search time
    
    # Start with various structured initializations for diversity
    population = []
    
    # Block patterns of various sizes
    for block_size in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64]:
        pattern = np.zeros(N, dtype=np.int8)
        for i in range(N):
            pattern[i] = (i // block_size) % 2
        population.append(pattern.copy())
    
    # Random initializations
    while len(population) < pop_size:
        population.append(np.random.randint(0, 2, N).astype(np.int8))
    
    # Convert to numpy array for vectorized ops
    population_arr = np.array(population, dtype=np.int8)
    
    # Evaluate initial population
    scores = count_mono_4ap_vectorized(population_arr)
    best_idx = np.argmin(scores)
    best_score = scores[best_idx]
    best_pop = population_arr[best_idx].copy()
    
    # Genetic algorithm
    mutation_rate = 0.05
    crossover_rate = 0.8
    
    gen = 0
    while time.time() < deadline - 10: # Leave 10s for final polish
        gen += 1
        # Sort by fitness
        order = np.argsort(scores)
        population_arr = population_arr[order]
        scores = scores[order]
        
        # Update best from current population
        if scores[0] < best_score:
            best_score = scores[0]
            best_pop = population_arr[0].copy()
            
        # Create next generation
        new_population = []
        
        # Elitism: keep top individuals
        elite_count = max(2, pop_size // 10)
        for i in range(elite_count):
            new_population.append(population_arr[i].copy())
        
        # Fill rest with offspring
        while len(new_population) < pop_size and time.time() < deadline - 10:
            # Tournament selection
            t1 = np.random.randint(0, pop_size)
            t2 = np.random.randint(0, pop_size)
            parent1 = population_arr[t1] if scores[t1] <= scores[t2] else population_arr[t2]
            
            t1 = np.random.randint(0, pop_size)
            t2 = np.random.randint(0, pop_size)
            parent2 = population_arr[t1] if scores[t1] <= scores[t2] else population_arr[t2]
            
            # Crossover (uniform)
            if np.random.random() < crossover_rate:
                mask = np.random.randint(0, 2, N).astype(np.int8)
                child = np.where(mask == 1, parent1, parent2).astype(np.int8)
            else:
                child = parent1.copy()
            
            # Mutation
            if np.random.random() < 0.9:
                num_mutations = np.random.poisson(N * mutation_rate)
                if num_mutations > 0:
                    mut_positions = np.random.choice(N, size=min(num_mutations, N), replace=False)
                    child[mut_positions] = 1 - child[mut_positions]
            
            new_population.append(child)
        
        # Evaluate new population
        population_arr = np.array(new_population, dtype=np.int8)
        scores = count_mono_4ap_vectorized(population_arr)
        
        # Periodically polish the best individual
        if gen % 5 == 0:
            polished_pop, polished_score = local_search(best_pop.copy(), max_passes=10)
            if polished_score < best_score:
                best_score = polished_score
                best_pop = polished_pop.copy()
                # Inject the polished individual into the population to spread its genes
                # Replace the worst individual
                worst_idx = np.argmax(scores)
                population_arr[worst_idx] = best_pop
                scores[worst_idx] = best_score
    
    # Final polish with more passes
    polished_pop, polished_score = local_search(best_pop.copy(), max_passes=100)
    if polished_score < best_score:
        best_score = polished_score
        best_pop = polished_pop.copy()
    
    # Convert to list of ints as required
    return [int(c) for c in best_pop]
