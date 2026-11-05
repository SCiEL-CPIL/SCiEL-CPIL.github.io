import numpy as np
import time
import random

N = 18

# EVOLVE-BLOCK-START
def construct():
    """Construct a sorting network for N wires using genetic algorithm search."""
    start_time = time.time()
    time_limit = 85.0  # Safety margin before 90s limit
    
    # Generate all 2^N binary inputs for 0-1 principle verification
    all_inputs = np.array(np.meshgrid(*[[0, 1]] * N), dtype=np.uint8).reshape(-1, N).T
    
    def verify_network(comparators):
        """Verify if a network sorts all binary inputs correctly."""
        if not comparators:
            return False
        # Check for duplicate comparators (not strictly necessary but good practice)
        # For speed, just simulate
        inputs = all_inputs.copy()
        for i, j in comparators:
            # Compare-exchange: min to i, max to j
            tmp = np.minimum(inputs[:, i], inputs[:, j])
            inputs[:, i] = tmp
            inputs[:, j] = np.maximum(inputs[:, i], inputs[:, j])
        # Check if all rows are sorted
        for k in range(N - 1):
            if np.any(inputs[:, k] > inputs[:, k + 1]):
                return False
        return True
    
    def count_comparators(comparators):
        return len(comparators)
    
    # Initial population: start with a known correct network
    # Bitonic sort network for 18 wires
    def bitonic_network(n):
        """Generate bitonic sort network for n wires (must be power of 2 or handle carefully)."""
        # For non-power-of-2, we'll use a simpler approach
        # Use odd-even transposition as seed but try to improve
        comparators = []
        # Phase 1: sort into bitonic sequence
        # Actually, let's just use a known small network structure
        
        # Simple approach: use odd-even transposition as starting point
        for phase in range(n):
            for i in range(phase % 2, n - 1, 2):
                comparators.append((i, i + 1))
        return comparators
    
    initial = bitonic_network(N)
    
    # Verify initial
    if not verify_network(initial):
        # Fallback: use a definitely correct network
        # Odd-even merge or just return a large network
        initial = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    best_network = list(initial)
    best_count = len(best_network)
    
    # Genetic algorithm
    population_size = 20
    mutation_rate = 0.3
    crossover_rate = 0.5
    
    # Initialize population with variations of the initial network
    population = [list(best_network)]
    for _ in range(population_size - 1):
        pop = list(best_network)
        # Random mutations
        for _ in range(random.randint(1, 5)):
            if random.random() < mutation_rate:
                # Add a random comparator
                i, j = random.sample(range(N), 2)
                if i > j:
                    i, j = j, i
                if (i, j) not in pop:
                    pop.append((i, j))
            elif random.random() < mutation_rate * 0.5 and len(pop) > 10:
                # Remove a random comparator
                idx = random.randint(0, len(pop) - 1)
                pop.pop(idx)
        population.append(pop)
    
    generation = 0
    while time.time() - start_time < time_limit:
        generation += 1
        new_population = []
        
        # Evaluate current population
        evaluated = []
        for pop in population:
            if verify_network(pop):
                evaluated.append((len(pop), pop))
        
        if not evaluated:
            # No valid networks in population, restart with initial
            population = [list(best_network)] + [list(best_network) for _ in range(population_size - 1)]
            continue
        
        # Sort by number of comparators (fewer is better)
        evaluated.sort(key=lambda x: x[0])
        
        # Update best
        if evaluated[0][0] < best_count:
            best_count = evaluated[0][0]
            best_network = list(evaluated[0][1])
        
        # Create new population
        # Elitism: keep the best
        new_population.append(list(evaluated[0][1]))
        
        # Crossover and mutation
        while len(new_population) < population_size:
            if time.time() - start_time > time_limit - 5:
                break
            
            # Select parents
            parent1 = random.choice(evaluated[:max(1, len(evaluated)//2)])[1]
            parent2 = random.choice(evaluated[:max(1, len(evaluated)//2)])[1]
            
            if random.random() < crossover_rate and len(parent1) > 0 and len(parent2) > 0:
                # Crossover: combine comparators from both parents
                child = list(parent1)
                # Add some from parent2
                for comp in parent2:
                    if comp not in child and random.random() < 0.3:
                        child.append(comp)
            else:
                child = list(parent1)
            
            # Mutation
            for _ in range(random.randint(1, 3)):
                r = random.random()
                if r < 0.4:
                    # Add comparator
                    i, j = random.sample(range(N), 2)
                    if i > j:
                        i, j = j, i
                    if (i, j) not in child:
                        child.append((i, j))
                elif r < 0.7 and len(child) > 5:
                    # Remove comparator
                    idx = random.randint(0, len(child) - 1)
                    child.pop(idx)
                else:
                    # Swap comparator
                    if len(child) > 0:
                        idx = random.randint(0, len(child) - 1)
                        i, j = random.sample(range(N), 2)
                        if i > j:
                            i, j = j, i
                        if (i, j) not in child:
                            child[idx] = (i, j)
            
            new_population.append(child)
        
        population = new_population
        
        # Early termination if we reach the theoretical lower bound
        if best_count <= 68:
            break
    
    return best_network
# EVOLVE-BLOCK-END
