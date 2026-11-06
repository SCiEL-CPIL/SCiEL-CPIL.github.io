"""Depth-by-depth genetic algorithm for sorting networks on 17 wires."""
import numpy as np
import random
import sys
import time

N = 17
MAX_COMPARATORS = 4096
TIME_LIMIT = 55  # seconds, leaving buffer for the 60s limit

def _verify_network(comparators):
    """
    Verify a sorting network using the Zero-One Principle with bit-vector simulation.
    Returns True if the network sorts all 2^N boolean inputs.
    """
    num_inputs = 1 << N
    state = np.zeros((num_inputs, N), dtype=np.uint8)
    for j in range(N):
        state[:, j] = ((np.arange(num_inputs, dtype=np.uint32) >> j) & 1).astype(np.uint8)
    
    for (i, j) in comparators:
        a = state[:, i]
        b = state[:, j]
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    
    diffs = state[:, :-1].astype(np.int8) - state[:, 1:].astype(np.int8)
    return np.all(diffs >= 0)

def _layers_to_comparators(layers):
    """Convert a list of layers (each a list of (i,j) pairs) to a flat list of comparators."""
    comparators = []
    for layer in layers:
        comparators.extend(layer)
    return comparators

def _comparators_to_layers(comparators):
    """Convert a flat list of comparators to layers based on which wires they use."""
    # This is a simple heuristic: group comparators that can be done in parallel
    # For simplicity, we'll just put each comparator in its own layer for now
    # A better approach would be to compute the depth
    return [[c] for c in comparators]

def _generate_random_layer():
    """Generate a random layer of non-overlapping comparators."""
    wires = list(range(N))
    random.shuffle(wires)
    layer = []
    i = 0
    while i < N - 1:
        if random.random() < 0.7:  # 70% chance to add a comparator
            a, b = wires[i], wires[i + 1]
            if a < b:
                layer.append((a, b))
            else:
                layer.append((b, a))
            i += 2
        else:
            i += 1
    return layer

def _generate_random_network(max_depth=15):
    """Generate a random network represented as a list of layers."""
    depth = random.randint(8, max_depth)
    layers = []
    for _ in range(depth):
        layers.append(_generate_random_layer())
    return layers

def _mutate_layers(layers, mutation_rate=0.1):
    """Mutate a network represented as layers."""
    layers = [list(layer) for layer in layers]
    n_mutations = int(len(layers) * mutation_rate) + random.randint(0, 2)
    
    for _ in range(n_mutations):
        op = random.choice(['add_comparator', 'remove_comparator', 'swap_wires', 'add_layer', 'remove_layer'])
        
        if op == 'add_comparator' and layers:
            # Add a comparator to a random layer if it doesn't conflict
            layer_idx = random.randint(0, len(layers) - 1)
            layer = layers[layer_idx]
            used_wires = set()
            for (a, b) in layer:
                used_wires.add(a)
                used_wires.add(b)
            available = [w for w in range(N) if w not in used_wires]
            if len(available) >= 2:
                a, b = random.sample(available, 2)
                if a < b:
                    layer.append((a, b))
                else:
                    layer.append((b, a))
        
        elif op == 'remove_comparator' and layers:
            # Remove a random comparator from a random layer
            layer_idx = random.randint(0, len(layers) - 1)
            layer = layers[layer_idx]
            if layer:
                idx = random.randint(0, len(layer) - 1)
                layer.pop(idx)
            # Remove empty layers
            layers = [l for l in layers if l]
        
        elif op == 'swap_wires' and layers:
            # Swap two wires in a random comparator
            layer_idx = random.randint(0, len(layers) - 1)
            layer = layers[layer_idx]
            if layer:
                comp_idx = random.randint(0, len(layer) - 1)
                a, b = layer[comp_idx]
                # Swap with another wire that's not used in this layer
                used_wires = set()
                for (x, y) in layer:
                    used_wires.add(x)
                    used_wires.add(y)
                available = [w for w in range(N) if w not in used_wires and w != a and w != b]
                if available:
                    new_wire = random.choice(available)
                    if new_wire < b:
                        layer[comp_idx] = (new_wire, b)
                    else:
                        layer[comp_idx] = (b, new_wire)
        
        elif op == 'add_layer' and len(layers) < 20:
            # Add a new random layer
            new_layer = _generate_random_layer()
            if new_layer:
                pos = random.randint(0, len(layers))
                layers.insert(pos, new_layer)
        
        elif op == 'remove_layer' and len(layers) > 1:
            # Remove a random layer
            idx = random.randint(0, len(layers) - 1)
            layers.pop(idx)
    
    return layers

def _crossover_layers(parent1, parent2):
    """Crossover: combine layers from two parents."""
    if not parent1 or not parent2:
        return [list(l) for l in parent1] if parent1 else [list(l) for l in parent2]
    
    # Find a crossover point
    max_len = max(len(parent1), len(parent2))
    point = random.randint(0, max_len)
    
    child = []
    for i in range(point):
        if i < len(parent1):
            child.append(list(parent1[i]))
    for i in range(point, len(parent2)):
        child.append(list(parent2[i]))
    
    return child

def _count_comparators(layers):
    """Count total comparators in a network represented as layers."""
    return sum(len(layer) for layer in layers)

def _evaluate_fitness(layers):
    """
    Fitness: lower is better.
    If the network is valid, fitness = number of comparators.
    If invalid, fitness = a large penalty.
    """
    comparators = _layers_to_comparators(layers)
    if len(comparators) > MAX_COMPARATORS:
        return MAX_COMPARATORS + 1000
    if _verify_network(comparators):
        return len(comparators)
    else:
        return MAX_COMPARATORS + 1000

def _odd_even_transposition_layers():
    """Generate the seed odd-even transposition network as layers."""
    layers = []
    for phase in range(N):
        layer = []
        for i in range(phase % 2, N - 1, 2):
            layer.append((i, i + 1))
        if layer:
            layers.append(layer)
    return layers

def _genetic_algorithm_depth_by_depth():
    """
    Run a genetic algorithm that optimizes comparator placement depth-by-depth.
    """
    start_time = time.time()
    
    # Initialize population with layer-based networks
    pop_size = 40
    population = []
    
    # Start with the seed network
    seed = _odd_even_transposition_layers()
    population.append(seed)
    
    # Add random networks
    for _ in range(pop_size - 1):
        population.append(_generate_random_network())
    
    # Evaluate initial population
    fitnesses = [_evaluate_fitness(ind) for ind in population]
    
    best_network = min(zip(fitnesses, population), key=lambda x: x[0])
    best_fitness = best_network[0]
    best_layers = best_network[1]
    
    print(f"Initial best: {best_fitness} comparators, {len(best_layers)} layers", file=sys.stderr)
    
    # Genetic algorithm loop
    generation = 0
    while time.time() - start_time < TIME_LIMIT:
        generation += 1
        
        # Selection: keep the best individuals
        sorted_pop = sorted(zip(fitnesses, population), key=lambda x: x[0])
        elite_size = max(2, pop_size // 4)
        new_population = [list(ind) for _, ind in sorted_pop[:elite_size]]
        
        # Generate offspring
        while len(new_population) < pop_size:
            if random.random() < 0.4 and len(population) >= 2:
                # Crossover
                p1 = random.choice(sorted_pop[:elite_size])[1]
                p2 = random.choice(sorted_pop[:elite_size])[1]
                child = _crossover_layers(p1, p2)
            else:
                # Mutation from a random parent
                parent = random.choice(sorted_pop[:elite_size])[1]
                child = _mutate_layers(parent, mutation_rate=0.08)
            
            new_population.append(child)
        
        population = new_population
        fitnesses = [_evaluate_fitness(ind) for ind in population]
        
        # Update best
        gen_best = min(zip(fitnesses, population), key=lambda x: x[0])
        if gen_best[0] < best_fitness:
            best_fitness = gen_best[0]
            best_layers = gen_best[1]
            print(f"Gen {generation}: new best {best_fitness} comparators, {len(best_layers)} layers", file=sys.stderr)
        
        # If we found a good solution, try to refine it with local search
        if best_fitness <= 80:
            for _ in range(3):
                mutated = _mutate_layers(best_layers, mutation_rate=0.03)
                fit = _evaluate_fitness(mutated)
                if fit < best_fitness:
                    best_fitness = fit
                    best_layers = mutated
                    print(f"Local search: new best {best_fitness} comparators", file=sys.stderr)
    
    return best_layers

def construct():
    """
    Construct a sorting network for 17 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    # Try the depth-by-depth genetic algorithm
    try:
        layers = _genetic_algorithm_depth_by_depth()
        comparators = _layers_to_comparators(layers)
        # Verify the result
        if _verify_network(comparators):
            return comparators
    except Exception as e:
        print(f"GA failed: {e}", file=sys.stderr)
    
    # Fallback to odd-even transposition
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
