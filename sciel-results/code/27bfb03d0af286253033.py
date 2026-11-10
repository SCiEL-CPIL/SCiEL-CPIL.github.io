"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def _count_mono_4ap_numpy(coloring):
    """Count monochromatic 4-term arithmetic progressions in a binary coloring of 1..256 using NumPy."""
    n = len(coloring)
    count = 0
    max_d = (n - 1) // 3
    c = np.array(coloring, dtype=np.int8)
    for d in range(1, max_d + 1):
        # Slices for a, a+d, a+2d, a+3d
        # Length of slice is n - 3*d
        # c[0 : n-3d]
        # c[d : n-2d]
        # c[2d : n-d]
        # c[3d : n]
        s1 = c[:n - 3 * d]
        s2 = c[d:n - 2 * d]
        s3 = c[2 * d:n - d]
        s4 = c[3 * d:n]
        # Check if all are equal
        # We can do: s1 == s2 == s3 == s4
        # But in numpy, this works element-wise.
        mono = (s1 == s2) & (s2 == s3) & (s3 == s4)
        count += np.sum(mono)
    return int(count)

def _local_search_sa(individual, iterations=200, temp_start=1.0, temp_end=0.01):
    """
    Perform a short Simulated Annealing local search on the individual.
    Returns the improved individual and its fitness.
    """
    current = list(individual)
    current_fit = _count_mono_4ap_numpy(current)
    
    n = len(current)
    # Estimate max possible improvement to scale temperature? 
    # Or just use fixed schedule.
    
    for i in range(iterations):
        # Linear cooling
        t = temp_start * (temp_end / temp_start) ** (i / iterations)
        
        # Pick a random position to flip
        idx = random.randint(0, n - 1)
        new_individual = current.copy()
        new_individual[idx] = 1 - new_individual[idx]
        
        new_fit = _count_mono_4ap_numpy(new_individual)
        
        delta = new_fit - current_fit
        if delta < 0:
            # Improvement
            current = new_individual
            current_fit = new_fit
        else:
            # Accept with probability
            if t > 0:
                prob = np.exp(-delta / t)
                if random.random() < prob:
                    current = new_individual
                    current_fit = new_fit
                    
    return current, current_fit

def construct():
    n = 256
    pop_size = 80
    generations = 100
    tournament_size = 5
    mutation_rate = 0.05
    
    # Time limit check
    start_time = time.time()
    time_limit = 110  # Leave 10s buffer
    
    random.seed(42)
    np.random.seed(42)
    
    population = []
    
    # Initialize with diverse structured patterns
    # 1. Alternating
    population.append([(i % 2) for i in range(n)])
    # 2. Blocks of 2
    population.append([(i // 2) % 2 for i in range(n)])
    # 3. Blocks of 3
    population.append([(i // 3) % 2 for i in range(n)])
    # 4. Blocks of 4
    population.append([(i // 4) % 2 for i in range(n)])
    # 5. Random
    for _ in range(pop_size - 4):
        population.append([random.randint(0, 1) for _ in range(n)])
    
    # Evaluate initial population
    fitnesses = [_count_mono_4ap_numpy(ind) for ind in population]
    
    best_individual = population[fitnesses.index(min(fitnesses))]
    best_fitness = min(fitnesses)
    
    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
            
        new_population = []
        
        # Elitism: Keep top 5
        sorted_indices = np.argsort(fitnesses)
        for idx in sorted_indices[:5]:
            new_population.append(list(population[idx]))
        
        while len(new_population) < pop_size:
            # Tournament selection
            parent1_idx = random.sample(range(pop_size), tournament_size)
            parent1_idx = min(parent1_idx, key=lambda i: fitnesses[i])
            parent1 = population[parent1_idx]
            
            parent2_idx = random.sample(range(pop_size), tournament_size)
            parent2_idx = min(parent2_idx, key=lambda i: fitnesses[i])
            parent2 = population[parent2_idx]
            
            # Uniform Crossover
            child = []
            for p1, p2 in zip(parent1, parent2):
                if random.random() < 0.5:
                    child.append(p1)
                else:
                    child.append(p2)
            
            # Mutation
            for i in range(n):
                if random.random() < mutation_rate:
                    child[i] = 1 - child[i]
            
            # Memetic Step: Local Search on the child
            # Only do local search on a subset to save time, or on all?
            # Let's do it on all children to ensure high quality, but limit iterations.
            # 50 iterations of SA is fast.
            child, child_fit = _local_search_sa(child, iterations=50, temp_start=1.0, temp_end=0.1)
            
            new_population.append(child)
        
        population = new_population
        fitnesses = [_count_mono_4ap_numpy(ind) for ind in population]
        
        # Track best
        current_best_idx = fitnesses.index(min(fitnesses))
        current_best_fitness = min(fitnesses)
        
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = list(population[current_best_idx])
            
            # Optional: Run a deeper local search on the global best occasionally
            if gen % 10 == 0:
                best_individual, best_fitness = _local_search_sa(best_individual, iterations=200, temp_start=1.0, temp_end=0.01)
    
    return best_individual
# EVOLVE-BLOCK-END
