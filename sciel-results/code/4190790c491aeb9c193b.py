"""Size-efficient sorting networks on 18 wires.
Uses a Genetic Algorithm (GA) to evolve sorting networks.
The fitness function is the number of unsorted Boolean states after applying the network.
Crossover and mutation operators are used to explore the space of comparator sequences.
"""

import time
import random
import numpy as np

N = 18
NUM_BOOLS = 1 << N  # 262144
DEADLINE = 85.0  # seconds (5s safety margin before 90s kill)


def _precompute_sorted_mask():
    """Precompute a boolean array: True if the state index is a sorted configuration."""
    mask = np.zeros(NUM_BOOLS, dtype=np.bool_)
    for k in range(N + 1):
        # k ones in the top k positions (wires N-k through N-1)
        # The sorted state for k ones is: 0...0 1...1 (k ones at the end)
        # Wait, the problem says "min to wire i and max to j".
        # Usually sorting networks sort in ascending order or descending order.
        # Let's check the seed program's definition.
        # Seed: val = ((1 << k) - 1) << (N - k)
        # If k=1, val = 1 << (N-1). Bit N-1 is set.
        # If we assume wire 0 is LSB and wire N-1 is MSB.
        # Sorted ascending: 0s at low indices, 1s at high indices.
        # So for k ones, the ones are at indices N-k, ..., N-1.
        # This matches the seed program.
        val = ((1 << k) - 1) << (N - k)
        mask[val] = True
    return mask


_SORTED_MASK = _precompute_sorted_mask()


def _odd_even_transposition_network():
    """Generate the odd-even transposition sorting network for N wires."""
    comps = []
    for phase in range(N):
        for i in range(phase % 2, N - 1, 2):
            comps.append((i, i + 1))
    return comps


def _apply_network_vectorized(comparators, initial_states):
    """Apply the network to a batch of initial states.
    initial_states: np.array of shape (num_states,) containing int32 values.
    Returns: np.array of final states.
    """
    current = initial_states.copy()
    for i, j in comparators:
        bit_i = (current >> i) & 1
        bit_j = (current >> j) & 1
        # Where bit_i=1 and bit_j=0, swap bits i and j
        # This effectively moves 1s towards higher indices (if i < j)
        mask_swap = (bit_i == 1) & (bit_j == 0)
        if np.any(mask_swap):
            clear_i = ~(1 << i) & 0xFFFF
            set_j = (1 << j)
            current = np.where(mask_swap, (current & clear_i) | set_j, current)
    return current


def _evaluate_fitness(comparators):
    """Calculate the number of unsorted states after applying the network.
    Lower is better. 0 means perfect sorting.
    """
    initial_states = np.arange(NUM_BOOLS, dtype=np.int32)
    final_states = _apply_network_vectorized(comparators, initial_states)
    # Count how many are NOT sorted
    return np.sum(~_SORTED_MASK[final_states])


def _verify_network(comparators):
    """Verify that the network sorts all 2^18 Boolean inputs.
    Returns True if valid, False otherwise.
    """
    return _evaluate_fitness(comparators) == 0


def _random_comparator():
    """Generate a random comparator (i, j) with i < j."""
    i = random.randint(0, N - 2)
    j = random.randint(i + 1, N - 1)
    return (i, j)


def _mutate(comparators, mutation_rate=0.1):
    """Apply mutation to a network.
    - With prob mutation_rate, replace a random comparator with a new random one.
    - With prob mutation_rate, insert a new random comparator at a random position.
    - With prob mutation_rate, delete a random comparator (if length > 1).
    """
    new_comps = list(comparators)
    if not new_comps:
        if random.random() < 0.5:
            new_comps.append(_random_comparator())
        return new_comps

    r = random.random()
    if r < mutation_rate:
        # Replace
        idx = random.randint(0, len(new_comps) - 1)
        new_comps[idx] = _random_comparator()
    elif r < 2 * mutation_rate:
        # Insert
        idx = random.randint(0, len(new_comps))
        new_comps.insert(idx, _random_comparator())
    elif r < 3 * mutation_rate:
        # Delete
        if len(new_comps) > 1:
            idx = random.randint(0, len(new_comps) - 1)
            new_comps.pop(idx)
    
    return new_comps


def _crossover(parent1, parent2):
    """Single-point crossover."""
    if not parent1 or not parent2:
        return parent1 if parent1 else parent2
    
    len1 = len(parent1)
    len2 = len(parent2)
    
    # Choose a crossover point based on lengths
    # To avoid bias, choose a point that is a fraction of the shorter parent's length
    # or just a random point in the range of the shorter one?
    # Let's pick a point in [0, min(len1, len2)]
    point = random.randint(0, min(len1, len2))
    
    # Child 1: head of P1, tail of P2
    child1 = parent1[:point] + parent2[point:]
    # Child 2: head of P2, tail of P1
    child2 = parent2[:point] + parent1[point:]
    
    return child1, child2


def _genetic_algorithm():
    """Run a genetic algorithm to find a small sorting network."""
    start_time = time.time()
    
    # Initialize population
    # Start with a mix of known good networks and random ones
    pop_size = 50
    population = []
    
    # Add a few seed networks (Odd-Even Transposition)
    seed_net = _odd_even_transposition_network()
    for _ in range(5):
        population.append(list(seed_net))
        
    # Add random networks of varying sizes
    for _ in range(pop_size - 5):
        size = random.randint(20, 100)
        net = []
        for _ in range(size):
            net.append(_random_comparator())
        population.append(net)
    
    # Evaluate initial population
    fitnesses = [_evaluate_fitness(net) for net in population]
    
    best_net = population[0]
    best_fitness = fitnesses[0]
    
    # Sort population by fitness
    pop_sorted = sorted(zip(fitnesses, population), key=lambda x: x[0])
    population = [p[1] for p in pop_sorted]
    fitnesses = [p[0] for p in pop_sorted]
    
    best_net = population[0]
    best_fitness = fitnesses[0]
    
    print(f"GA Init: Best fitness = {best_fitness}, Size = {len(best_net)}")
    
    generation = 0
    while time.time() - start_time < DEADLINE - 5:
        generation += 1
        new_population = []
        
        # Elitism: keep the best few
        elite_count = 5
        for i in range(elite_count):
            new_population.append(population[i])
        
        # Fill the rest with offspring
        while len(new_population) < pop_size:
            # Tournament selection
            t_size = 5
            candidates = random.sample(range(len(population)), t_size)
            winner_idx = min(candidates, key=lambda i: fitnesses[i])
            parent1 = population[winner_idx]
            
            # Second parent
            candidates2 = random.sample(range(len(population)), t_size)
            winner_idx2 = min(candidates2, key=lambda i: fitnesses[i])
            parent2 = population[winner_idx2]
            
            # Crossover
            child1, child2 = _crossover(parent1, parent2)
            
            # Mutation
            child1 = _mutate(child1, mutation_rate=0.2)
            child2 = _mutate(child2, mutation_rate=0.2)
            
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)
        
        # Evaluate new population
        new_fitnesses = [_evaluate_fitness(net) for net in new_population]
        
        # Update best
        min_idx = np.argmin(new_fitnesses)
        if new_fitnesses[min_idx] < best_fitness:
            best_fitness = new_fitnesses[min_idx]
            best_net = new_population[min_idx]
            print(f"GA Gen {generation}: New Best fitness = {best_fitness}, Size = {len(best_net)}")
            
        # Sort new population
        pop_sorted = sorted(zip(new_fitnesses, new_population), key=lambda x: x[0])
        population = [p[1] for p in pop_sorted]
        fitnesses = [p[0] for p in pop_sorted]
        
        # Early stop if we found a perfect network
        if best_fitness == 0:
            break
            
    return best_net


def construct():
    """Construct a sorting network for 18 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    start_time = time.time()
    
    # Start with a good seed
    seed_net = _odd_even_transposition_network()
    
    # Run GA
    ga_net = _genetic_algorithm()
    
    # Compare with seed
    if len(ga_net) < len(seed_net):
        best_net = ga_net
    else:
        best_net = seed_net
        
    # Final verification
    if not _verify_network(best_net):
        # Fallback to seed
        best_net = seed_net
        
    return best_net

