"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random

def _count_mono_4ap(coloring):
    """Count monochromatic 4-term arithmetic progressions in a binary coloring of 1..256."""
    n = len(coloring)
    count = 0
    # Precompute for efficiency
    # For each a and d, check if a, a+d, a+2d, a+3d are all in range and same color
    # a ranges from 0 to n-1 (0-indexed), d >= 1
    # a+3d < n => a < n - 3d => d < n/3
    max_d = (n - 1) // 3
    for d in range(1, max_d + 1):
        for a in range(n - 3 * d):
            c0 = coloring[a]
            if c0 == coloring[a + d] == coloring[a + 2 * d] == coloring[a + 3 * d]:
                count += 1
    return count

def _crossover(parent1, parent2):
    """Uniform crossover."""
    child = []
    for p1, p2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child.append(p1)
        else:
            child.append(p2)
    return child

def _mutate(individual, mutation_rate=0.02):
    """Bit-flip mutation."""
    mutated = list(individual)
    for i in range(len(mutated)):
        if random.random() < mutation_rate:
            mutated[i] = 1 - mutated[i]
    return mutated

def _evaluate_population(population):
    """Evaluate all individuals in the population."""
    fitnesses = [_count_mono_4ap(ind) for ind in population]
    return fitnesses

def construct():
    n = 256
    pop_size = 50
    generations = 200
    tournament_size = 5
    mutation_rate = 0.03
    
    # Initialize population with diverse random individuals
    # Include some structured initial individuals
    random.seed(42)
    np.random.seed(42)
    
    population = []
    
    # Add the seed solution
    population.append([(i // 2) % 2 for i in range(n)])
    
    # Add some structured patterns
    # Blocks of 1
    population.append([i % 2 for i in range(n)])
    # Blocks of 3
    population.append([(i // 3) % 2 for i in range(n)])
    # Blocks of 4
    population.append([(i // 4) % 2 for i in range(n)])
    # Random
    for _ in range(pop_size - 4):
        population.append([random.randint(0, 1) for _ in range(n)])
    
    # Evaluate initial population
    fitnesses = _evaluate_population(population)
    
    best_individual = population[fitnesses.index(min(fitnesses))]
    best_fitness = min(fitnesses)
    
    # Genetic algorithm loop
    for gen in range(generations):
        # Create new population
        new_population = []
        
        # Elitism: keep the best individual
        best_idx = fitnesses.index(min(fitnesses))
        new_population.append(list(population[best_idx]))
        
        while len(new_population) < pop_size:
            # Tournament selection
            parent1_idx = random.sample(range(pop_size), tournament_size)
            parent1_idx = min(parent1_idx, key=lambda i: fitnesses[i])
            parent1 = population[parent1_idx]
            
            parent2_idx = random.sample(range(pop_size), tournament_size)
            parent2_idx = min(parent2_idx, key=lambda i: fitnesses[i])
            parent2 = population[parent2_idx]
            
            # Crossover
            child = _crossover(parent1, parent2)
            
            # Mutation
            child = _mutate(child, mutation_rate)
            
            new_population.append(child)
        
        population = new_population
        fitnesses = _evaluate_population(population)
        
        # Track best
        current_best_idx = fitnesses.index(min(fitnesses))
        current_best_fitness = min(fitnesses)
        
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_individual = list(population[current_best_idx])
    
    return best_individual
# EVOLVE-BLOCK-END
