"""Rewritten program: genetic algorithm for sorting networks on 17 wires."""
import numpy as np
import random
import sys

N = 17
MAX_COMPARATORS = 4096
TIME_LIMIT = 55  # seconds, leaving buffer for the 60s limit

def _verify_network(comparators):
    """
    Verify a sorting network using the Zero-One Principle with bit-vector simulation.
    Returns True if the network sorts all 2^N boolean inputs.
    """
    # Represent all 2^N boolean inputs as a matrix of shape (2^N, N)
    # For efficiency, we use numpy bit operations.
    # Each row is a boolean input vector.
    # We simulate the network on all inputs simultaneously.
    
    num_inputs = 1 << N
    # Create the initial state: all possible boolean vectors
    # Using integer representation: each input is an integer from 0 to 2^N - 1
    # We need to apply comparators to all inputs at once.
    
    # State: array of shape (num_inputs, N) with 0s and 1s
    # Initialize: input i has bit j set if (i >> j) & 1
    state = np.zeros((num_inputs, N), dtype=np.uint8)
    for j in range(N):
        # Bit j of each input
        state[:, j] = ((np.arange(num_inputs, dtype=np.uint32) >> j) & 1).astype(np.uint8)
    
    for (i, j) in comparators:
        # Compare-exchange: min to i, max to j
        # For each input, compare state[k, i] and state[k, j]
        a = state[:, i]
        b = state[:, j]
        # min = a & b, max = a | b for boolean values
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    
    # Check if sorted: for each input, state[k, :] should be non-increasing
    # i.e., state[k, 0] >= state[k, 1] >= ... >= state[k, N-1]
    # This is equivalent to: no pair (k, j) where j < N-1 and state[k, j] < state[k, j+1]
    diffs = state[:, :-1].astype(np.int8) - state[:, 1:].astype(np.int8)
    # If any diff is negative, not sorted
    return np.all(diffs >= 0)

def _count_comparators(comparators):
    return len(comparators)

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _generate_random_network(num_comparators):
    """Generate a random network with a given number of comparators."""
    comparators = []
    for _ in range(num_comparators):
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        comparators.append((i, j))
    return comparators

def _crossover(parent1, parent2):
    """Crossover: take a random prefix from parent1 and suffix from parent2."""
    if not parent1 or not parent2:
        return list(parent1) if parent1 else list(parent2)
    point = random.randint(0, min(len(parent1), len(parent2)))
    child = list(parent1[:point]) + list(parent2[point:])
    return child

def _mutate(comparators, mutation_rate=0.1):
    """Mutate a network: add, remove, or swap comparators."""
    comparators = list(comparators)
    n_mutations = int(len(comparators) * mutation_rate) + random.randint(0, 2)
    for _ in range(n_mutations):
        op = random.choice(['add', 'remove', 'swap', 'reorder'])
        if op == 'add' and len(comparators) < MAX_COMPARATORS:
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            pos = random.randint(0, len(comparators))
            comparators.insert(pos, (i, j))
        elif op == 'remove' and len(comparators) > 1:
            pos = random.randint(0, len(comparators) - 1)
            comparators.pop(pos)
        elif op == 'swap' and len(comparators) >= 2:
            i, j = random.sample(range(len(comparators)), 2)
            comparators[i], comparators[j] = comparators[j], comparators[i]
        elif op == 'reorder' and len(comparators) >= 2:
            i, j = random.sample(range(len(comparators)), 2)
            comparators[i], comparators[j] = comparators[j], comparators[i]
    return comparators

def _evaluate_fitness(comparators):
    """
    Fitness: lower is better.
    If the network is valid, fitness = number of comparators.
    If invalid, fitness = a large penalty.
    """
    if len(comparators) > MAX_COMPARATORS:
        return MAX_COMPARATORS + 1000
    if _verify_network(comparators):
        return len(comparators)
    else:
        return MAX_COMPARATORS + 1000

def _genetic_algorithm():
    """
    Run a genetic algorithm to find a small sorting network.
    """
    import time
    start_time = time.time()
    
    # Initialize population
    pop_size = 30
    population = []
    
    # Start with the seed network
    seed = _odd_even_transposition()
    population.append(seed)
    
    # Add some random networks
    for _ in range(pop_size - 1):
        # Start with a smaller random network and grow
        num_comp = random.randint(40, 80)
        population.append(_generate_random_network(num_comp))
    
    # Evaluate initial population
    fitnesses = [_evaluate_fitness(ind) for ind in population]
    
    best_network = min(zip(fitnesses, population), key=lambda x: x[0])
    best_fitness = best_network[0]
    best_comparators = best_network[1]
    
    print(f"Initial best: {best_fitness} comparators", file=sys.stderr)
    
    # Genetic algorithm loop
    generation = 0
    while time.time() - start_time < TIME_LIMIT:
        generation += 1
        
        # Selection: keep the best individuals
        sorted_pop = sorted(zip(fitnesses, population), key=lambda x: x[0])
        elite_size = max(1, pop_size // 5)
        new_population = [ind for _, ind in sorted_pop[:elite_size]]
        
        # Generate offspring
        while len(new_population) < pop_size:
            if random.random() < 0.3 and len(population) >= 2:
                # Crossover
                p1 = random.choice(sorted_pop[:elite_size])[1]
                p2 = random.choice(sorted_pop[:elite_size])[1]
                child = _crossover(p1, p2)
            else:
                # Mutation from a random parent
                parent = random.choice(sorted_pop[:elite_size])[1]
                child = _mutate(parent, mutation_rate=0.05)
            
            new_population.append(child)
        
        population = new_population
        fitnesses = [_evaluate_fitness(ind) for ind in population]
        
        # Update best
        gen_best = min(zip(fitnesses, population), key=lambda x: x[0])
        if gen_best[0] < best_fitness:
            best_fitness = gen_best[0]
            best_comparators = gen_best[1]
            print(f"Gen {generation}: new best {best_fitness} comparators", file=sys.stderr)
        
        # If we found a good solution, try to refine it
        if best_fitness <= 75:
            # Try local search
            for _ in range(5):
                mutated = _mutate(best_comparators, mutation_rate=0.02)
                fit = _evaluate_fitness(mutated)
                if fit < best_fitness:
                    best_fitness = fit
                    best_comparators = mutated
                    print(f"Local search: new best {best_fitness} comparators", file=sys.stderr)
    
    return best_comparators

def construct():
    """
    Construct a sorting network for 17 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    # Try the genetic algorithm
    try:
        network = _genetic_algorithm()
        # Verify the result
        if _verify_network(network):
            return network
    except Exception as e:
        print(f"GA failed: {e}", file=sys.stderr)
    
    # Fallback to odd-even transposition
    return _odd_even_transposition()
