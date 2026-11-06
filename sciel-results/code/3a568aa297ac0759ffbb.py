"""Improved program: genetic algorithm with depth-based generation for sorting networks on 17 wires."""
import numpy as np
import random
import sys
import time

N = 17
MAX_COMPARATORS = 4096
TIME_LIMIT = 55  # seconds, leaving buffer for the 60s limit


def _verify_network(comparators):
    """
    Verify a sorting network using the Zero-One Principle with efficient bit-vector simulation.
    Returns True if the network sorts all 2^N boolean inputs.
    Uses a packed representation: each bit position represents one wire across all inputs.
    """
    if len(comparators) == 0:
        return N <= 1
    
    # Pack all 2^N inputs into N bit-vectors, each of length 2^N
    # bit_vector[w] has bit k set if input k has 1 on wire w
    num_inputs = 1 << N
    # Create bit vectors for each wire
    # For wire w, bit k is set if the k-th input has value 1 on wire w
    # Input k is the integer k, so bit w of input k is (k >> w) & 1
    
    # Build bit vectors using numpy for speed
    inputs = np.arange(num_inputs, dtype=np.uint32)
    
    # State: list of N uint32 arrays, each of length num_inputs//32 (packed)
    # Actually, let's use a different approach: keep as (num_inputs, N) but use uint8
    # and vectorize the comparator application
    
    state = np.zeros((num_inputs, N), dtype=np.uint8)
    for w in range(N):
        state[:, w] = ((inputs >> w) & 1).astype(np.uint8)
    
    for (i, j) in comparators:
        a = state[:, i]
        b = state[:, j]
        # min = a & b, max = a | b
        state[:, i] = a & b
        state[:, j] = a | b
    
    # Check sorted: state[:, w] >= state[:, w+1] for all w
    # i.e., state[:, w] - state[:, w+1] >= 0 for all w
    for w in range(N - 1):
        if np.any(state[:, w] < state[:, w + 1]):
            return False
    return True


def _verify_network_fast(comparators):
    """
    Faster verification using a different strategy:
    Track the set of possible outputs after each comparator.
    """
    if len(comparators) == 0:
        return N <= 1
    
    # Use the zero-one principle: check all 2^N boolean inputs
    # But do it more efficiently by tracking which inputs are already sorted
    
    num_inputs = 1 << N
    inputs = np.arange(num_inputs, dtype=np.uint32)
    
    state = np.zeros((num_inputs, N), dtype=np.uint8)
    for w in range(N):
        state[:, w] = ((inputs >> w) & 1).astype(np.uint8)
    
    for (i, j) in comparators:
        a = state[:, i]
        b = state[:, j]
        # Only update where a > b (i.e., a=1, b=0)
        swap_mask = (a == 1) & (b == 0)
        if np.any(swap_mask):
            state[swap_mask, i] = 0
            state[swap_mask, j] = 1
    
    # Check sorted
    for w in range(N - 1):
        if np.any(state[:, w] < state[:, w + 1]):
            return False
    return True


def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]


def _generate_depth_based_network(num_layers, comparators_per_layer=None):
    """
    Generate a network with a specified number of layers.
    Each layer contains a set of non-conflicting comparators (each wire used at most once).
    """
    if comparators_per_layer is None:
        comparators_per_layer = random.randint(1, N // 2)
    
    comparators = []
    for _ in range(num_layers):
        # Generate a set of non-conflicting comparators for this layer
        used_wires = set()
        layer_comparators = []
        attempts = 0
        while len(layer_comparators) < comparators_per_layer and attempts < 50:
            attempts += 1
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            if i not in used_wires and j not in used_wires:
                used_wires.add(i)
                used_wires.add(j)
                layer_comparators.append((i, j))
        comparators.extend(layer_comparators)
    
    return comparators


def _remove_redundant(comparators):
    """
    Try to remove redundant comparators from a valid network.
    Greedily tries to remove each comparator and checks if the network is still valid.
    """
    if not _verify_network(comparators):
        return comparators
    
    result = list(comparators)
    i = 0
    while i < len(result):
        # Try removing comparator at position i
        candidate = result[:i] + result[i+1:]
        if _verify_network(candidate):
            result = candidate
            # Don't increment i, check the same position again
        else:
            i += 1
    
    return result


def _mutate_network(comparators, mutation_type=None):
    """
    Mutate a network with various operations.
    """
    comparators = list(comparators)
    if mutation_type is None:
        mutation_type = random.choice(['remove', 'add', 'swap', 'move', 'replace'])
    
    if mutation_type == 'remove' and len(comparators) > 1:
        pos = random.randint(0, len(comparators) - 1)
        comparators.pop(pos)
    elif mutation_type == 'add' and len(comparators) < MAX_COMPARATORS:
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        pos = random.randint(0, len(comparators))
        comparators.insert(pos, (i, j))
    elif mutation_type == 'swap' and len(comparators) >= 2:
        i, j = random.sample(range(len(comparators)), 2)
        comparators[i], comparators[j] = comparators[j], comparators[i]
    elif mutation_type == 'move' and len(comparators) >= 2:
        i, j = random.sample(range(len(comparators)), 2)
        comp = comparators.pop(i)
        comparators.insert(j, comp)
    elif mutation_type == 'replace' and len(comparators) > 0:
        pos = random.randint(0, len(comparators) - 1)
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        comparators[pos] = (i, j)
    
    return comparators


def _genetic_algorithm():
    """
    Run a genetic algorithm to find a small sorting network.
    Uses depth-based generation and focused mutation.
    """
    start_time = time.time()
    
    # Initialize population with diverse starting points
    pop_size = 20
    population = []
    
    # Start with the seed network
    seed = _odd_even_transposition()
    population.append(seed)
    
    # Try to reduce the seed by removing redundant comparators
    reduced_seed = _remove_redundant(seed)
    population.append(reduced_seed)
    
    # Add depth-based networks with varying parameters
    for _ in range(pop_size - 2):
        num_layers = random.randint(8, 16)
        comps_per_layer = random.randint(2, 8)
        net = _generate_depth_based_network(num_layers, comps_per_layer)
        population.append(net)
    
    # Evaluate initial population
    fitnesses = []
    for ind in population:
        if _verify_network(ind):
            fitnesses.append(len(ind))
        else:
            fitnesses.append(MAX_COMPARATORS + 1000)
    
    best_idx = np.argmin(fitnesses)
    best_fitness = fitnesses[best_idx]
    best_comparators = population[best_idx]
    
    print(f"Initial best: {best_fitness} comparators", file=sys.stderr)
    
    # Genetic algorithm loop
    generation = 0
    stagnant_generations = 0
    best_ever = best_fitness
    
    while time.time() - start_time < TIME_LIMIT:
        generation += 1
        
        # Selection: keep the best individuals
        sorted_indices = np.argsort(fitnesses)
        elite_size = max(1, pop_size // 5)
        new_population = [population[i] for i in sorted_indices[:elite_size]]
        
        # Generate offspring
        while len(new_population) < pop_size:
            r = random.random()
            if r < 0.2 and len(population) >= 2:
                # Crossover: take prefix from one, suffix from another
                p1_idx = random.choice(sorted_indices[:elite_size])
                p2_idx = random.choice(sorted_indices[:elite_size])
                p1 = population[p1_idx]
                p2 = population[p2_idx]
                point = random.randint(0, min(len(p1), len(p2)))
                child = list(p1[:point]) + list(p2[point:])
            elif r < 0.5:
                # Mutation from elite
                parent_idx = random.choice(sorted_indices[:elite_size])
                parent = population[parent_idx]
                child = _mutate_network(parent)
            elif r < 0.7:
                # Depth-based generation with random parameters
                num_layers = random.randint(8, 16)
                comps_per_layer = random.randint(2, 8)
                child = _generate_depth_based_network(num_layers, comps_per_layer)
            else:
                # Mutation from any individual
                parent = random.choice(population)
                child = _mutate_network(parent)
            
            new_population.append(child)
        
        population = new_population
        fitnesses = []
        for ind in population:
            if _verify_network(ind):
                fitnesses.append(len(ind))
            else:
                fitnesses.append(MAX_COMPARATORS + 1000)
        
        # Update best
        gen_best_idx = np.argmin(fitnesses)
        if fitnesses[gen_best_idx] < best_fitness:
            best_fitness = fitnesses[gen_best_idx]
            best_comparators = population[gen_best_idx]
            stagnant_generations = 0
            print(f"Gen {generation}: new best {best_fitness} comparators", file=sys.stderr)
            
            # Try to further reduce by removing redundant comparators
            reduced = _remove_redundant(best_comparators)
            if len(reduced) < best_fitness:
                best_fitness = len(reduced)
                best_comparators = reduced
                print(f"Reduction: {best_fitness} comparators", file=sys.stderr)
        else:
            stagnant_generations += 1
        
        # If stagnant for a while, inject more diversity
        if stagnant_generations > 10:
            # Add some fresh depth-based networks
            for _ in range(5):
                num_layers = random.randint(8, 16)
                comps_per_layer = random.randint(2, 8)
                population.append(_generate_depth_based_network(num_layers, comps_per_layer))
            # Re-evaluate
            fitnesses = []
            for ind in population:
                if _verify_network(ind):
                    fitnesses.append(len(ind))
                else:
                    fitnesses.append(MAX_COMPARATORS + 1000)
            stagnant_generations = 0
    
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
