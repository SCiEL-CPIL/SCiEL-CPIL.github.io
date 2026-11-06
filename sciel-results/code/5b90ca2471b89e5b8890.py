"""Rewritten program: Genetic Algorithm for sorting networks on 17 wires with layered structure."""
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
    # Initialize state with all boolean vectors
    for j in range(N):
        state[:, j] = ((np.arange(num_inputs, dtype=np.uint32) >> j) & 1).astype(np.uint8)
    
    for (i, j) in comparators:
        a = state[:, i]
        b = state[:, j]
        state[:, i] = np.minimum(a, b)
        state[:, j] = np.maximum(a, b)
    
    # Check if sorted: non-increasing
    diffs = state[:, :-1].astype(np.int8) - state[:, 1:].astype(np.int8)
    return np.all(diffs >= 0)

def _odd_even_transposition():
    """Generate the seed odd-even transposition network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def _flatten(layers):
    """Flatten a list of layers into a list of comparators."""
    comps = []
    for layer in layers:
        comps.extend(layer)
    return comps

def _is_valid_layer(layer):
    """Check if a layer is valid (no overlapping wires)."""
    wires = []
    for i, j in layer:
        if i in wires or j in wires:
            return False
        wires.append(i)
        wires.append(j)
    return True

def _random_layer():
    """Generate a random valid layer."""
    available = list(range(N))
    random.shuffle(available)
    layer = []
    i = 0
    while i < len(available) - 1:
        # Pick two wires from available
        # To keep it simple, just take pairs from the shuffled list
        # This ensures no overlap
        if i + 1 < len(available):
            layer.append((available[i], available[i+1]))
            i += 2
        else:
            break
    return layer

def _generate_random_network(max_depth=10):
    """Generate a random layered network."""
    layers = []
    for _ in range(max_depth):
        layers.append(_random_layer())
    return layers

def _crossover_layers(layers1, layers2):
    """Crossover: combine layers from parent1 and parent2."""
    if not layers1 or not layers2:
        return list(layers1) if layers1 else list(layers2)
    
    # Random point crossover
    point = random.randint(0, min(len(layers1), len(layers2)))
    child = layers1[:point] + layers2[point:]
    return [list(layer) for layer in child] # Deep copy

def _mutate_layers(layers, mutation_rate=0.1):
    """Mutate a layered network."""
    layers = [list(layer) for layer in layers] # Deep copy
    n_mutations = max(1, int(len(layers) * mutation_rate))
    
    for _ in range(n_mutations):
        op = random.choice(['add', 'remove', 'move'])
        
        if op == 'add':
            # Add a random comparator to a random valid layer
            if layers:
                idx = random.randint(0, len(layers) - 1)
                # Find available wires in this layer
                used = set()
                for i, j in layers[idx]:
                    used.add(i)
                    used.add(j)
                available = [w for w in range(N) if w not in used]
                if len(available) >= 2:
                    i, j = random.sample(available, 2)
                    if i > j: i, j = j, i
                    layers[idx].append((i, j))
        
        elif op == 'remove':
            # Remove a random comparator from a random non-empty layer
            non_empty = [i for i, l in enumerate(layers) if l]
            if non_empty:
                idx = random.choice(non_empty)
                comp_idx = random.randint(0, len(layers[idx]) - 1)
                layers[idx].pop(comp_idx)
                if not layers[idx]:
                    layers.pop(idx) # Remove empty layer
        
        elif op == 'move':
            # Move a comparator from one layer to another
            non_empty = [i for i, l in enumerate(layers) if l]
            if len(non_empty) >= 2:
                src_idx = random.choice(non_empty)
                # Choose a destination layer
                dest_candidates = [i for i in range(len(layers)) if i != src_idx]
                if not dest_candidates:
                    continue
                dest_idx = random.choice(dest_candidates)
                
                comp_idx = random.randint(0, len(layers[src_idx]) - 1)
                comp = layers[src_idx].pop(comp_idx)
                
                # Check if we can add it to dest layer
                used = set()
                for i, j in layers[dest_idx]:
                    used.add(i)
                    used.add(j)
                if comp[0] not in used and comp[1] not in used:
                    layers[dest_idx].append(comp)
                else:
                    # Put it back if it doesn't fit
                    layers[src_idx].append(comp)
                    
    return layers

def _evaluate_fitness(layers):
    """
    Fitness: lower is better.
    If the network is valid, fitness = number of comparators.
    If invalid, fitness = a large penalty.
    """
    comps = _flatten(layers)
    if len(comps) > MAX_COMPARATORS:
        return MAX_COMPARATORS + 1000
    if _verify_network(comps):
        return len(comps)
    else:
        return MAX_COMPARATORS + 1000

def _local_search(layers, time_limit=5.0):
    """
    Try to remove redundant comparators from a valid network.
    """
    start_time = time.time()
    current_layers = [list(layer) for layer in layers]
    current_comps = _flatten(current_layers)
    
    if not _verify_network(current_comps):
        return current_layers
        
    improved = True
    while improved and (time.time() - start_time) < time_limit:
        improved = False
        # Try removing each comparator
        for i in range(len(current_layers)):
            if not current_layers[i]:
                continue
            for j in range(len(current_layers[i])):
                # Try removing comparator (i, j)
                comp = current_layers[i][j]
                # Temporarily remove
                current_layers[i].pop(j)
                if not current_layers[i]:
                    current_layers.pop(i)
                    # Adjust indices if necessary, but we break loop anyway
                    new_comps = _flatten(current_layers)
                    if _verify_network(new_comps):
                        improved = True
                        print(f"Local search: removed {comp}, size {len(new_comps)}", file=sys.stderr)
                        # Reset loop to restart from beginning as indices changed
                        break
                    else:
                        # Put it back
                        if i < len(current_layers):
                            current_layers[i].append(comp)
                        else:
                            current_layers.append([comp])
                        # If we popped the layer, we need to be careful. 
                        # Simpler: just keep the layer if it becomes empty? No, empty layers are harmless but add nothing.
                        # Let's just put it back and continue
                    continue
                new_comps = _flatten(current_layers)
                if _verify_network(new_comps):
                    improved = True
                    print(f"Local search: removed {comp}, size {len(new_comps)}", file=sys.stderr)
                    # Restart inner loop
                    break
                else:
                    # Put it back
                    current_layers[i].insert(j, comp)
            if improved:
                break
    return current_layers

def _genetic_algorithm():
    """
    Run a genetic algorithm to find a small sorting network.
    """
    start_time = time.time()
    
    # Initialize population with layered networks
    pop_size = 20
    population = []
    
    # Seed
    seed_comps = _odd_even_transposition()
    # Convert seed to layers (Odd-even transposition is already layered by phase)
    seed_layers = []
    for phase in range(N):
        layer = []
        for i in range(phase % 2, N - 1, 2):
            layer.append((i, i + 1))
        seed_layers.append(layer)
    population.append(seed_layers)
    
    # Random diverse starts
    for _ in range(pop_size - 1):
        max_depth = random.randint(8, 15)
        population.append(_generate_random_network(max_depth))
    
    # Evaluate initial population
    fitnesses = [_evaluate_fitness(ind) for ind in population]
    
    best_network = min(zip(fitnesses, population), key=lambda x: x[0])
    best_fitness = best_network[0]
    best_layers = best_network[1]
    
    print(f"Initial best: {best_fitness} comparators", file=sys.stderr)
    
    # Genetic algorithm loop
    generation = 0
    while time.time() - start_time < TIME_LIMIT - 10: # Leave time for local search
        generation += 1
        
        # Selection
        sorted_pop = sorted(zip(fitnesses, population), key=lambda x: x[0])
        elite_size = max(1, pop_size // 5)
        new_population = [ind for _, ind in sorted_pop[:elite_size]]
        
        # Generate offspring
        while len(new_population) < pop_size:
            if random.random() < 0.5 and len(population) >= 2:
                p1 = random.choice(sorted_pop[:elite_size])[1]
                p2 = random.choice(sorted_pop[:elite_size])[1]
                child = _crossover_layers(p1, p2)
            else:
                parent = random.choice(sorted_pop[:elite_size])[1]
                child = _mutate_layers(parent, mutation_rate=0.1)
            
            new_population.append(child)
        
        population = new_population
        fitnesses = [_evaluate_fitness(ind) for ind in population]
        
        gen_best = min(zip(fitnesses, population), key=lambda x: x[0])
        if gen_best[0] < best_fitness:
            best_fitness = gen_best[0]
            best_layers = gen_best[1]
            print(f"Gen {generation}: new best {best_fitness} comparators", file=sys.stderr)
    
    # Final Local Search on the best candidate
    if best_fitness < MAX_COMPARATORS + 1000: # If we found a valid network
        print("Starting local search...", file=sys.stderr)
        best_layers = _local_search(best_layers, time_limit=5.0)
        final_comps = _flatten(best_layers)
        if _verify_network(final_comps):
            best_fitness = len(final_comps)
            print(f"Final best: {best_fitness} comparators", file=sys.stderr)
            return final_comps
    
    return _flatten(best_layers)

def construct():
    """
    Construct a sorting network for 17 wires.
    Returns an ordered list of integer pairs (i, j).
    """
    try:
        network = _genetic_algorithm()
        if _verify_network(network):
            return network
    except Exception as e:
        print(f"GA failed: {e}", file=sys.stderr)
    
    return _odd_even_transposition()
