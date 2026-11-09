"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

N = 18
NUM_INPUTS = 1 << N  # 262144

# EVOLVE-BLOCK-START

def _apply_comparator_bitparallel(comparators, inputs):
    """
    Apply a list of comparators to a set of inputs.
    inputs: array of shape (NUM_INPUTS, N) with uint8 values (0/1).
    comparators: list of (i, j) pairs.
    Returns the sorted (or partially sorted) array.
    """
    if len(comparators) == 0:
        return inputs.copy()
    
    # We process the comparators one by one.
    # To be efficient, we could batch them if they are independent, but for simplicity 
    # and correctness in the GA, we just apply them sequentially.
    # However, 153 comparators * 262144 inputs might be slow in pure Python loops.
    # Let's use vectorized operations for each comparator step.
    
    # Actually, applying 153 comparators sequentially to 262144x18 array:
    # Each step: 
    #   a = inputs[:, i]
    #   b = inputs[:, j]
    #   min_ab = np.minimum(a, b)
    #   max_ab = np.maximum(a, b)
    #   inputs[:, i] = min_ab
    #   inputs[:, j] = max_ab
    # This is 153 * 6 vectorized ops on arrays of size 262144.
    # This should be fast enough in NumPy (milliseconds per network evaluation).
    
    res = inputs.copy()
    for i, j in comparators:
        a = res[:, i]
        b = res[:, j]
        # min and max
        # Since values are 0/1, we can use arithmetic or np.minimum/maximum
        # np.minimum/maximum is fine.
        res[:, i] = np.minimum(a, b)
        res[:, j] = np.maximum(a, b)
    return res

def _is_sorted_network(comparators):
    """
    Check if the network sorts all 2^N binary inputs.
    """
    # Generate all 2^N binary inputs
    # We need an array of shape (2^N, N)
    # Input k (0 to 2^N - 1) has bits corresponding to its binary representation.
    # We need to be careful about bit ordering.
    # Let's say input value v has bits b_0 ... b_{N-1} where b_0 is LSB.
    # Wire i corresponds to bit i.
    
    # Efficient way to generate all inputs:
    # np.arange(2**N) -> shape (2**N,)
    # We need to expand this to (2**N, N)
    # bits = (inputs[:, None] >> np.arange(N)[None, :]) & 1
    
    # But 2**18 is 262144.
    # inputs = np.arange(2**N, dtype=np.uint32)
    # bits = (inputs[:, None] >> np.arange(N)[None, :]) & 1
    
    # This creates a 262144 x 18 array.
    # Memory: 262144 * 18 * 4 bytes (uint32) ~ 18 MB. This is fine.
    # Using uint8 is better: 262144 * 18 * 1 ~ 4.7 MB.
    
    inputs = np.arange(1 << N, dtype=np.uint32)
    # Shift and mask
    # np.arange(N) is [0, 1, ..., 17]
    # We want bit i to be at column i.
    # inputs[:, None] is (262144, 1)
    # np.arange(N)[None, :] is (1, 18)
    # Broadcasting works.
    bits = ((inputs[:, None] >> np.arange(N)[None, :]) & 1).astype(np.uint8)
    
    sorted_bits = _apply_comparator_bitparallel(comparators, bits)
    
    # Check if sorted_bits[:, i] <= sorted_bits[:, i+1] for all i
    # We can check if the array is non-decreasing along axis 1.
    # diff = np.diff(sorted_bits, axis=1)
    # If all diffs >= 0, it's sorted.
    
    # Alternatively, check if sorted_bits is equal to the sorted version of bits.
    # But bits are already 0/1. The sorted version of a binary vector is just 
    # zeros followed by ones.
    # So for each row, the number of 1s is preserved, and they should be at the end.
    
    # Let's check the non-decreasing property.
    diffs = np.diff(sorted_bits, axis=1)
    return np.all(diffs >= 0)

def _count_unsorted(comparators, inputs=None):
    """
    Return the number of unsorted inputs (0-1 vectors) for the given network.
    """
    if inputs is None:
        inputs = np.arange(1 << N, dtype=np.uint32)
        bits = ((inputs[:, None] >> np.arange(N)[None, :]) & 1).astype(np.uint8)
    else:
        bits = inputs
        
    sorted_bits = _apply_comparator_bitparallel(comparators, bits)
    diffs = np.diff(sorted_bits, axis=1)
    # A row is unsorted if any diff < 0
    unsorted_rows = np.any(diffs < 0, axis=1)
    return np.sum(unsorted_rows)

def _odd_even_transposition_network():
    """
    Generate the odd-even transposition sorting network.
    """
    comparators = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comparators.append((i, i + 1))
    return comparators

def _random_network(num_comparators):
    """
    Generate a random sorting network with the given number of comparators.
    """
    comparators = []
    for _ in range(num_comparators):
        i = random.randint(0, N - 1)
        j = random.randint(0, N - 1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        comparators.append((i, j))
    return comparators

def _crossover(parent1, parent2):
    """
    Uniform crossover: for each position, pick from parent1 or parent2.
    """
    child = []
    for c1, c2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child.append(c1)
        else:
            child.append(c2)
    return child

def _mutation(network, mutation_rate=0.1):
    """
    Mutate a network by randomly changing, adding, or removing comparators.
    """
    new_network = network.copy()
    num_mutations = int(len(new_network) * mutation_rate)
    
    for _ in range(num_mutations):
        mutation_type = random.choice(['change', 'add', 'remove'])
        
        if mutation_type == 'change' and len(new_network) > 0:
            idx = random.randint(0, len(new_network) - 1)
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i != j:
                if i > j:
                    i, j = j, i
                new_network[idx] = (i, j)
                
        elif mutation_type == 'add':
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if i != j:
                if i > j:
                    i, j = j, i
                # Insert at random position
                idx = random.randint(0, len(new_network))
                new_network.insert(idx, (i, j))
                
        elif mutation_type == 'remove' and len(new_network) > 0:
            idx = random.randint(0, len(new_network) - 1)
            new_network.pop(idx)
            
    return new_network

def construct():
    """
    Construct a sorting network using a genetic algorithm.
    """
    start_time = time.time()
    deadline = start_time + 85  # 85 seconds to be safe within 90s limit
    
    # Initialize population with the odd-even transposition network
    base_network = _odd_even_transposition_network()
    
    # Create initial population
    population_size = 50
    population = [base_network.copy() for _ in range(population_size)]
    
    # Add some random networks to the population
    for i in range(population_size // 2):
        num_comp = random.randint(60, 120)
        population.append(_random_network(num_comp))
    
    population = population[:population_size]
    
    best_network = base_network.copy()
    best_unsorted = _count_unsorted(best_network)
    
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Evaluate population
        fitnesses = []
        for network in population:
            unsorted = _count_unsorted(network)
            fitnesses.append(unsorted)
            if unsorted < best_unsorted:
                best_unsorted = unsorted
                best_network = network.copy()
                if best_unsorted == 0:
                    return best_network
        
        # Sort by fitness (lower is better)
        sorted_population = sorted(zip(population, fitnesses), key=lambda x: x[1])
        population = [net for net, fit in sorted_population]
        fitnesses = [fit for net, fit in sorted_population]
        
        # Selection: keep top half
        elite_size = population_size // 2
        new_population = [net.copy() for net in population[:elite_size]]
        
        # Crossover and mutation to fill the rest
        while len(new_population) < population_size and time.time() < deadline:
            # Tournament selection
            p1_idx = random.randint(0, elite_size - 1)
            p2_idx = random.randint(0, elite_size - 1)
            
            # Crossover
            if random.random() < 0.8:
                child = _crossover(population[p1_idx], population[p2_idx])
            else:
                child = population[p1_idx].copy()
            
            # Mutation
            child = _mutation(child, mutation_rate=0.15)
            
            new_population.append(child)
        
        population = new_population[:population_size]
        
        # If we've been stuck for a while, try a different approach
        if generation % 10 == 0 and best_unsorted > 0:
            # Try to add some random comparators to the best network
            for _ in range(5):
                modified = best_network.copy()
                num_adds = random.randint(1, 5)
                for _ in range(num_adds):
                    i = random.randint(0, N - 1)
                    j = random.randint(0, N - 1)
                    if i != j:
                        if i > j:
                            i, j = j, i
                        idx = random.randint(0, len(modified))
                        modified.insert(idx, (i, j))
                
                unsorted = _count_unsorted(modified)
                if unsorted < best_unsorted:
                    best_unsorted = unsorted
                    best_network = modified
                    if best_unsorted == 0:
                        return best_network

    return best_network

# EVOLVE-BLOCK-END
