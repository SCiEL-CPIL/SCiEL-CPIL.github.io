import numpy as np

N = 50

def _fitness(f):
    f = np.asarray(f, dtype=float)
    f = np.maximum(f, 0)
    l1 = np.sum(f)
    if l1 < 1e-12:
        return 0.0
    f = f / l1
    l2_sq = np.sum(f**2)
    conv = np.convolve(f, f, mode='full')
    amax = np.max(conv)
    if amax < 1e-12:
        return 0.0
    return l2_sq / amax

def construct():
    pop_size = 200
    generations = 100
    mutation_rate = 0.1
    mutation_scale = 0.5
    tournament_size = 5
    
    # Initialize population with random non-negative values
    # Mix of uniform, random spikes, and smooth randoms to ensure diversity
    population = []
    for _ in range(pop_size):
        if np.random.rand() < 0.3:
            # Random uniform
            f = np.random.rand(N)
        elif np.random.rand() < 0.6:
            # Random smooth (sum of few gaussians)
            f = np.zeros(N)
            for _ in range(np.random.randint(1, 4)):
                center = np.random.randint(0, N)
                sigma = np.random.uniform(1, N/2)
                x = np.arange(N)
                f += np.exp(-0.5 * ((x - center) / sigma) ** 2)
        else:
            # Sparse spikes
            f = np.zeros(N)
            num_spikes = np.random.randint(1, 5)
            indices = np.random.choice(N, num_spikes, replace=False)
            for idx in indices:
                f[idx] = np.random.rand()
        
        # Normalize to L1=1 for consistent comparison, but keep raw for mutation
        # Actually, fitness is scale-invariant, so we can just evaluate as is
        population.append(f)
    
    best_f = None
    best_score = -np.inf
    
    for gen in range(generations):
        # Evaluate fitness
        scores = np.array([_fitness(f) for f in population])
        
        # Track best
        max_idx = np.argmax(scores)
        if scores[max_idx] > best_score:
            best_score = scores[max_idx]
            best_f = population[max_idx].copy()
            
        # Selection and Crossover
        new_population = []
        
        # Elitism: keep top 10%
        elite_count = max(1, pop_size // 10)
        elite_indices = np.argsort(scores)[-elite_count:]
        for idx in elite_indices:
            new_population.append(population[idx].copy())
            
        # Fill rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            parent1_idx = np.random.choice(pop_size, tournament_size)
            parent1 = population[np.argmax(scores[parent1_idx])]
            
            parent2_idx = np.random.choice(pop_size, tournament_size)
            parent2 = population[np.argmax(scores[parent2_idx])]
            
            # Uniform Crossover
            mask = np.random.rand(N) > 0.5
            child = np.where(mask, parent1, parent2)
            
            # Mutation
            # Add Gaussian noise
            noise = np.random.normal(0, mutation_scale, N)
            child = child + noise
            
            # Clip to non-negative
            child = np.maximum(child, 0)
            
            # Ensure not all zeros
            if np.sum(child) < 1e-12:
                child = np.random.rand(N)
                
            new_population.append(child)
            
        population = new_population
        
    # Return the best found, normalized to L1=1 (or just the vector, as score is scale invariant)
    # The task says "sequence of nonnegative step heights".
    # The evaluator calculates the score. The score is invariant to scaling.
    # So we can return the vector as is, or normalized. Let's normalize for cleanliness.
    if best_f is None:
        return list(np.ones(N))
    
    l1 = np.sum(best_f)
    if l1 > 1e-12:
        best_f = best_f / l1
        
    return list(best_f)
