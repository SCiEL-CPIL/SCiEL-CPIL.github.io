"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

import time
import random
import numpy as np

# EVOLVE-BLOCK-START

def _generate_bitonic_network(n):
    """Generate Bitonic Sort network for n wires."""
    def _bitonic_sort(start, length, direction):
        if length <= 1:
            return []
        half = length // 2
        network = []
        network.extend(_bitonic_sort(start, half, 1))
        network.extend(_bitonic_sort(start + half, half, -1))
        network.extend(_bitonic_merge(start, length, 1))
        return network

    def _bitonic_merge(start, length, direction):
        if length <= 1:
            return []
        half = length // 2
        network = []
        for i in range(half):
            left = start + i
            right = start + i + half
            if direction == 1:
                network.append((left, right))
            else:
                network.append((right, left))
        network.extend(_bitonic_merge(start, half, direction))
        network.extend(_bitonic_merge(start + half, half, direction))
        return network

    return _bitonic_sort(0, n, 1)

def _verify_network_fast(network):
    """
    Fast verification using zero-one principle with bit-parallelism.
    Returns 0 if network is correct, else number of unsorted inputs.
    """
    # Pack all 2^16 inputs into 32-bit integers (one per wire)
    # Wire k gets bit k of each mask, packed into 64-bit words
    # We'll use numpy uint64 arrays, each holding 64 inputs per element
    # 65536 inputs / 64 = 1024 uint64 values per wire
    
    # Generate all 65536 masks as uint64
    all_masks = np.arange(1 << N, dtype=np.uint64)
    # Reshape to (1024, 64) where each row is 64 consecutive masks
    all_masks_2d = all_masks.reshape(1024, 64)
    
    # Extract wire k: bit k of each mask
    # For wire k, the value at position p in a 64-element group is (mask >> k) & 1
    # We can compute this via: (all_masks_2d >> k) & 1, but we need to handle the 64-bit packing
    # Actually, let's just use the simple approach with uint8 for clarity and speed
    # 65536 x 16 = 1M bytes, very manageable
    
    inputs_arr = ((all_masks[:, None] >> np.arange(N, dtype=np.uint64)[None, :]) & 1).astype(np.uint8)
    
    for i, j in network:
        # Compare-exchange: min to i, max to j
        # Swap if inputs_arr[:, i] > inputs_arr[:, j]
        mask = inputs_arr[:, i] > inputs_arr[:, j]
        if np.any(mask):
            tmp = inputs_arr[mask, i].copy()
            inputs_arr[mask, i] = inputs_arr[mask, j]
            inputs_arr[mask, j] = tmp
    
    # Check if sorted (non-decreasing)
    diffs = np.diff(inputs_arr, axis=1)
    unsorted_rows = np.any(diffs < 0, axis=1)
    return int(np.sum(unsorted_rows))

def _apply_layer(layer, state):
    """Apply a layer (list of disjoint comparators) to a state array."""
    for i, j in layer:
        mask = state[:, i] > state[:, j]
        if np.any(mask):
            tmp = state[mask, i].copy()
            state[mask, i] = state[mask, j]
            state[mask, j] = tmp
    return state

def _evaluate_fitness(layers):
    """
    Evaluate fitness of a layer sequence.
    Returns (is_valid, violation_count).
    """
    # Generate all 2^16 zero-one inputs
    all_masks = np.arange(1 << N, dtype=np.int32)
    state = ((all_masks[:, None] >> np.arange(N)[None, :]) & 1).astype(np.int8)
    
    for layer in layers:
        for i, j in layer:
            mask = state[:, i] > state[:, j]
            if np.any(mask):
                tmp = state[mask, i].copy()
                state[mask, i] = state[mask, j]
                state[mask, j] = tmp
    
    # Check if sorted
    diffs = np.diff(state, axis=1)
    unsorted_rows = np.any(diffs < 0, axis=1)
    violations = int(np.sum(unsorted_rows))
    
    if violations == 0:
        return (True, 0)
    else:
        return (False, violations)

def _random_layer():
    """Generate a random layer of 8 disjoint comparators."""
    wires = list(range(N))
    random.shuffle(wires)
    layer = []
    for k in range(N // 2):
        i = wires[2 * k]
        j = wires[2 * k + 1]
        if i > j:
            i, j = j, i
        layer.append((i, j))
    return layer

def _ga_search(initial_layers, time_limit):
    """
    Genetic algorithm to optimize layer sequence.
    """
    start_time = time.time()
    
    # Initial population
    pop_size = 50
    population = [list(initial_layers)]
    
    # Add random initial individuals
    for _ in range(pop_size - 1):
        num_layers = random.randint(5, 12)
        ind = [_random_layer() for _ in range(num_layers)]
        population.append(ind)
    
    best_individual = None
    best_fitness = float('inf')
    best_is_valid = False
    
    # Evaluate initial population
    for ind in population:
        is_valid, violations = _evaluate_fitness(ind)
        # Fitness: prefer valid, then fewer violations, then fewer total comparators
        total_comps = sum(len(layer) for layer in ind)
        if is_valid:
            fitness = total_comps
        else:
            fitness = 10000 + violations + total_comps
        
        if fitness < best_fitness:
            best_fitness = fitness
            best_individual = [layer[:] for layer in ind]
            best_is_valid = is_valid
    
    generation = 0
    while time.time() - start_time < time_limit:
        generation += 1
        
        # Selection: tournament selection
        def select_tournament(pop, k=3):
            candidates = random.sample(pop, k)
            best = None
            best_fit = float('inf')
            for c in candidates:
                is_valid, violations = _evaluate_fitness(c)
                total_comps = sum(len(layer) for layer in c)
                if is_valid:
                    fit = total_comps
                else:
                    fit = 10000 + violations + total_comps
                if fit < best_fit:
                    best_fit = fit
                    best = c
            return best
        
        # Crossover and mutation
        new_population = []
        
        # Keep best individual (elitism)
        new_population.append([layer[:] for layer in best_individual])
        
        while len(new_population) < pop_size:
            parent1 = select_tournament(population)
            parent2 = select_tournament(population)
            
            # Crossover: swap layers at random point
            child = []
            if len(parent1) <= 1 or len(parent2) <= 1:
                child = [layer[:] for layer in parent1]
            else:
                point = random.randint(1, min(len(parent1), len(parent2)) - 1)
                child = [layer[:] for layer in parent1[:point]] + [layer[:] for layer in parent2[point:]]
            
            # Mutation
            for m in range(3):
                if random.random() < 0.3:
                    # Replace a random layer
                    if len(child) > 0:
                        idx = random.randint(0, len(child) - 1)
                        child[idx] = _random_layer()
                    else:
                        child.append(_random_layer())
                elif random.random() < 0.2 and len(child) > 0:
                    # Add a layer
                    child.append(_random_layer())
                elif random.random() < 0.2 and len(child) > 1:
                    # Remove a layer
                    idx = random.randint(0, len(child) - 1)
                    child.pop(idx)
            
            new_population.append(child)
        
        population = new_population
        
        # Evaluate new population and update best
        for ind in population:
            is_valid, violations = _evaluate_fitness(ind)
            total_comps = sum(len(layer) for layer in ind)
            if is_valid:
                fitness = total_comps
            else:
                fitness = 10000 + violations + total_comps
            
            if fitness < best_fitness:
                best_fitness = fitness
                best_individual = [layer[:] for layer in ind]
                best_is_valid = is_valid
    
    return best_individual, best_is_valid, best_fitness

def _layers_to_network(layers):
    """Convert layer sequence to flat network list."""
    network = []
    for layer in layers:
        network.extend(layer)
    return network

def _construct_impl():
    start_time = time.time()
    DEADLINE = 85  # seconds, with safety margin for 90s kill
    
    # Start with Bitonic Sort (119 comparators for n=16)
    best_net = _generate_bitonic_network(N)
    
    # Verify initial network
    if _verify_network_fast(best_net) != 0:
        # Fallback to odd-even transposition
        best_net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
    
    # Local search: aggressively try removing comparators
    # Use a greedy approach: repeatedly try to remove random comparators
    random.seed(42)
    attempts = 0
    while time.time() - start_time < DEADLINE - 5:
        attempts += 1
        if len(best_net) <= 55:
            break
        # Try removing a random comparator
        idx = random.randint(0, len(best_net) - 1)
        candidate = best_net[:idx] + best_net[idx+1:]
        if _verify_network_fast(candidate) == 0:
            best_net = candidate
        # Also try swapping adjacent comparators
        if random.random() < 0.3 and len(best_net) >= 2:
            idx = random.randint(0, len(best_net) - 2)
            candidate = best_net[:idx] + [best_net[idx+1], best_net[idx]] + best_net[idx+2:]
            if _verify_network_fast(candidate) == 0:
                best_net = candidate
    
    # Try adding comparators in strategic positions to enable further removals
    remaining_time = DEADLINE - (time.time() - start_time)
    if remaining_time > 3:
        for _ in range(20):
            if time.time() - start_time > DEADLINE - 5:
                break
            # Try adding a random comparator
            i = random.randint(0, N - 2)
            j = random.randint(i + 1, N - 1)
            idx = random.randint(0, len(best_net))
            candidate = best_net[:idx] + [(i, j)] + best_net[idx:]
            # Now try to remove some comparators
            for _ in range(10):
                if time.time() - start_time > DEADLINE - 5:
                    break
                ridx = random.randint(0, len(candidate) - 1)
                reduced = candidate[:ridx] + candidate[ridx+1:]
                if _verify_network_fast(reduced) == 0:
                    candidate = reduced
            if len(candidate) < len(best_net):
                best_net = candidate
    
    return best_net

def construct():
    try:
        return _construct_impl()
    except Exception:
        return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

# EVOLVE-BLOCK-END
