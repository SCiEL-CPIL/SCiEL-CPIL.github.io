"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np

# EVOLVE-BLOCK-START
def _fitness(lst):
    """
    Compute the number of distinct d > 0 such that a, a+d, a+2d are all in the set.
    Uses bitmask for efficient evaluation.
    """
    mask = 0
    for x in lst:
        mask |= (1 << (x - 1))
    
    count = 0
    # Max d is 255
    for d in range(1, 256):
        # Check if intersection of mask, mask>>d, mask>>2d is non-empty
        if (mask & (mask >> d) & (mask >> (2 * d))) != 0:
            count += 1
    return count

def _random_individual():
    """Generate a random 32-element subset of 1..512."""
    return random.sample(range(1, 513), 32)

def _crossover(parent1, parent2):
    """
    Uniform crossover on lists of 32 integers.
    """
    child = []
    for i in range(32):
        if random.random() < 0.5:
            child.append(parent1[i])
        else:
            child.append(parent2[i])
    
    # Handle duplicates: if any duplicates exist, replace them with random unique values
    # This is rare if parents are distinct, but crossover can produce duplicates
    # Actually, uniform crossover on lists of distinct items can produce duplicates.
    # We need to ensure the child has 32 distinct integers.
    
    # A better way: 
    # 1. Create a set of the child so far.
    # 2. If len < 32, fill with random unique values from 1..512 not in set.
    # 3. If len > 32 (impossible if we pick 1 from each pos), wait.
    # If we pick 1 item from each of 32 positions, we get 32 items.
    # If there are duplicates, we have fewer than 32 unique items.
    
    child_set = set(child)
    if len(child_set) < 32:
        # Find missing count
        needed = 32 - len(child_set)
        # Get all available numbers
        # This is expensive if done by iterating 1..512.
        # But we can just pick random numbers until we have enough.
        # Or, simpler: just regenerate the child if duplicates? 
        # Or fix duplicates.
        
        # Let's just replace duplicates with random unique values.
        # First, identify unique items and their indices? 
        # Actually, let's just rebuild the child to ensure uniqueness.
        
        # Strategy: 
        # 1. Take unique elements from child.
        # 2. Fill the rest with random elements from 1..512 not in the unique set.
        
        unique_child = list(child_set)
        # We need 32 - len(unique_child) more elements.
        # To avoid slow "available" list generation, we pick randoms and check.
        
        # Since the set is small, collisions are unlikely.
        while len(unique_child) < 32:
            new_val = random.randint(1, 512)
            if new_val not in child_set:
                child_set.add(new_val)
                unique_child.append(new_val)
        
        # Now we have 32 unique values. Shuffle to break parent structure slightly?
        # Or just return as is. The order doesn't matter for fitness.
        return unique_child
    else:
        return child

def _mutate(individual, mutation_rate=0.1):
    """
    Mutation: replace random elements with new random unique values.
    """
    individual = list(individual)
    ind_set = set(individual)
    
    # Number of mutations
    num_mutations = int(32 * mutation_rate)
    # Ensure at least 1 mutation if rate > 0? 
    # Let's just do per-element probability as before, but optimized.
    
    for i in range(32):
        if random.random() < mutation_rate:
            old_val = individual[i]
            ind_set.remove(old_val)
            
            while True:
                new_val = random.randint(1, 512)
                if new_val not in ind_set:
                    ind_set.add(new_val)
                    individual[i] = new_val
                    break
                    
    return individual

def _local_search(individual, max_iters=50):
    """
    Hill climbing: try swapping one element at a time.
    """
    best_ind = list(individual)
    best_fit = _fitness(best_ind)
    ind_set = set(best_ind)
    
    for _ in range(max_iters):
        improved = False
        # Try a few random swaps
        for _ in range(20):
            i = random.randint(0, 31)
            
            # Pick a new value not in set
            while True:
                new_val = random.randint(1, 512)
                if new_val not in ind_set:
                    break
            
            # Swap
            old_val = best_ind[i]
            ind_set.remove(old_val)
            ind_set.add(new_val)
            best_ind[i] = new_val
            
            new_fit = _fitness(best_ind)
            
            if new_fit > best_fit:
                best_fit = new_fit
                improved = True
            else:
                # Revert
                ind_set.remove(new_val)
                ind_set.add(old_val)
                best_ind[i] = old_val
        
        if not improved:
            break
            
    return best_ind, best_fit

def construct():
    """
    Run a genetic algorithm with local search refinement.
    """
    random.seed(42)
    
    population_size = 60
    generations = 300
    elite_size = 10
    mutation_rate = 0.15
    
    # Initialize population
    population = [_random_individual() for _ in range(population_size)]
    fitnesses = [_fitness(ind) for ind in population]
    
    best_idx = max(range(population_size), key=lambda i: fitnesses[i])
    best_individual = population[best_idx]
    best_fitness = fitnesses[best_idx]
    
    for gen in range(generations):
        # Sort population by fitness
        paired = sorted(zip(fitnesses, population), key=lambda x: -x[0])
        fitnesses = [p[0] for p in paired]
        population = [p[1] for p in paired]
        
        # Local search on the best individual of this generation
        ls_individual, ls_fitness = _local_search(population[0], max_iters=20)
        if ls_fitness > best_fitness:
            best_fitness = ls_fitness
            best_individual = ls_individual
            # Inject the improved individual into the population
            population[0] = ls_individual
            fitnesses[0] = ls_fitness
        
        # Keep elite
        new_population = [list(population[i]) for i in range(elite_size)]
        
        # Generate offspring
        while len(new_population) < population_size:
            # Tournament selection
            def tournament():
                indices = random.sample(range(len(population)), 3)
                best_idx_t = max(indices, key=lambda i: fitnesses[i])
                return population[best_idx_t]
            
            parent1 = tournament()
            parent2 = tournament()
            child = _crossover(parent1, parent2)
            child = _mutate(child, mutation_rate)
            new_population.append(child)
        
        population = new_population
        fitnesses = [_fitness(ind) for ind in population]
        
        # Update best from population if local search didn't find something better
        current_best_idx = max(range(population_size), key=lambda i: fitnesses[i])
        if fitnesses[current_best_idx] > best_fitness:
            best_fitness = fitnesses[current_best_idx]
            best_individual = population[current_best_idx]
            
        # Early stopping if we hit a theoretical max? 
        # Max possible d is 255. So max score is 255.
        if best_fitness == 255:
            break
    
    return sorted(best_individual)

# EVOLVE-BLOCK-END
