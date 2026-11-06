"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import numpy as np
import time

N = 17

def _apply_comparator(state, i, j):
    """Apply a compare-exchange to a state vector (in-place)."""
    if state[i] > state[j]:
        state[i], state[j] = state[j], state[i]
    return state

def _evaluate_network(comparators, samples=None, sample_size=2048):
    """
    Evaluate a sorting network using the zero-one principle with random Boolean inputs.
    Returns True if the network sorts all sampled inputs.
    """
    if samples is None:
        samples = np.random.randint(0, 2, size=(sample_size, N))
    
    for sample in samples:
        state = sample.copy()
        for i, j in comparators:
            _apply_comparator(state, i, j)
        if not np.all(state[:-1] <= state[1:]):
            return False
    return True

def _evaluate_network_full(comparators):
    """
    Evaluate a sorting network over ALL 2^17 Boolean inputs (zero-one principle).
    Returns True if the network sorts all inputs.
    """
    for mask in range(1 << N):
        state = np.array([(mask >> i) & 1 for i in range(N)], dtype=np.int8)
        for i, j in comparators:
            _apply_comparator(state, i, j)
        if not np.all(state[:-1] <= state[1:]):
            return False
    return True

def _compute_depth(comparators):
    """Compute the parallel depth of the network."""
    if not comparators:
        return 0
    level = [0] * N
    depth = 0
    for i, j in comparators:
        lvl = max(level[i], level[j]) + 1
        level[i] = lvl
        level[j] = lvl
        depth = max(depth, lvl)
    return depth

def _layers_to_comparators(layers):
    """Convert list of layers (each a list of (i,j)) to flat comparator list."""
    comps = []
    for layer in layers:
        for comp in layer:
            comps.append(comp)
    return comps

def _comparators_to_layers(comparators):
    """Convert flat comparator list to layers based on wire levels."""
    level = [0] * N
    layers = {}
    for i, j in comparators:
        lvl = max(level[i], level[j]) + 1
        level[i] = lvl
        level[j] = lvl
        if lvl not in layers:
            layers[lvl] = []
        layers[lvl].append((i, j))
    
    max_lvl = max(layers.keys()) if layers else 0
    result = []
    for l in range(1, max_lvl + 1):
        result.append(layers.get(l, []))
    return result

def _is_valid_layer(layer):
    """Check if a layer is a valid matching (no wire appears more than once)."""
    used = set()
    for i, j in layer:
        if i in used or j in used:
            return False
        used.add(i)
        used.add(j)
    return True

def _generate_seed_layers():
    """Generate odd-even transposition network as layers."""
    layers = []
    for phase in range(N):
        layer = [(i, i + 1) for i in range(phase % 2, N - 1, 2)]
        layers.append(layer)
    return layers

def _mutate_layers(layers, mutation_rate=0.1):
    """
    Apply mutations to the layer representation.
    Mutations:
    - Remove a random comparator from a random layer
    - Swap two comparators within the same layer
    - Swap a comparator between two layers
    - Insert a new random comparator into a random layer (if valid)
    """
    new_layers = [list(layer) for layer in layers]
    num_comps = sum(len(layer) for layer in new_layers)
    num_mutations = max(1, int(mutation_rate * num_comps))
    
    for _ in range(num_mutations):
        op = random.choice(['remove', 'swap_within', 'swap_between', 'insert'])
        
        if op == 'remove' and num_comps > 0:
            # Pick a random comparator
            total = sum(len(layer) for layer in new_layers)
            if total == 0:
                continue
            pick = random.randint(0, total - 1)
            for li, layer in enumerate(new_layers):
                if pick < len(layer):
                    new_layers[li].pop(pick)
                    break
                pick -= len(layer)
        
        elif op == 'swap_within':
            # Find a layer with at least 2 comparators
            candidates = [li for li, layer in enumerate(new_layers) if len(layer) >= 2]
            if candidates:
                li = random.choice(candidates)
                layer = new_layers[li]
                i, j = random.sample(range(len(layer)), 2)
                layer[i], layer[j] = layer[j], layer[i]
        
        elif op == 'swap_between':
            # Find two non-empty layers
            non_empty = [li for li, layer in enumerate(new_layers) if len(layer) > 0]
            if len(non_empty) >= 2:
                li1, li2 = random.sample(non_empty, 2)
                i1 = random.randint(0, len(new_layers[li1]) - 1)
                i2 = random.randint(0, len(new_layers[li2]) - 1)
                comp1 = new_layers[li1][i1]
                comp2 = new_layers[li2][i2]
                # Try the swap
                new_layers[li1][i1] = comp2
                new_layers[li2][i2] = comp1
                # Check validity
                if not _is_valid_layer(new_layers[li1]) or not _is_valid_layer(new_layers[li2]):
                    # Revert
                    new_layers[li1][i1] = comp1
                    new_layers[li2][i2] = comp2
        
        elif op == 'insert':
            # Pick a random layer
            li = random.randint(0, len(new_layers) - 1)
            # Pick a random comparator
            i, j = random.sample(range(N), 2)
            if i > j:
                i, j = j, i
            # Check if valid to add
            test_layer = new_layers[li] + [(i, j)]
            if _is_valid_layer(test_layer):
                new_layers[li].append((i, j))
    
    return new_layers

def _crossover_layers(layers1, layers2):
    """
    Crossover: take a random split point in terms of total comparators.
    """
    comps1 = _layers_to_comparators(layers1)
    comps2 = _layers_to_comparators(layers2)
    
    if not comps1 or not comps2:
        return _comparators_to_layers(comps1 if comps1 else comps2)
    
    split = random.randint(0, max(len(comps1), len(comps2)))
    child_comps = comps1[:split] + comps2[split:]
    return _comparators_to_layers(child_comps)

def _local_search_layers(layers, max_iters=50, sample_size=512):
    """
    Local search on layer representation: try removing redundant comparators.
    """
    new_layers = [list(layer) for layer in layers]
    changed = True
    iters = 0
    while changed and iters < max_iters:
        changed = False
        iters += 1
        # Try removing each comparator
        for li in range(len(new_layers) - 1, -1, -1):
            for ci in range(len(new_layers[li]) - 1, -1, -1):
                # Temporarily remove
                removed = new_layers[li].pop(ci)
                comps = _layers_to_comparators(new_layers)
                if _evaluate_network(comps, sample_size=sample_size):
                    changed = True
                    break
                else:
                    # Restore
                    new_layers[li].insert(ci, removed)
            if changed:
                break
    return new_layers

def construct():
    """
    Construct a sorting network using a genetic algorithm with layer-based representation.
    Returns a list of (i, j) comparator pairs.
    """
    start_time = time.time()
    max_time = 55
    
    # Initialize with seed network in layer form
    best_layers = _generate_seed_layers()
    best_comps = _layers_to_comparators(best_layers)
    best_len = len(best_comps)
    
    # Population
    pop_size = 20
    population = [best_layers]
    for _ in range(pop_size - 1):
        mutated = _mutate_layers(best_layers, mutation_rate=0.2)
        population.append(mutated)
    
    generation = 0
    while time.time() - start_time < max_time:
        generation += 1
        
        # Evaluate and select
        scored = []
        for layers in population:
            comps = _layers_to_comparators(layers)
            if _evaluate_network(comps, sample_size=1024):
                score = -len(comps)
            else:
                score = -10000
            scored.append((score, layers))
        
        scored.sort(key=lambda x: x[0])
        
        # Update best
        if scored[0][0] > -10000:
            best_comps_candidate = _layers_to_comparators(scored[0][1])
            if len(best_comps_candidate) < best_len:
                if _evaluate_network_full(best_comps_candidate):
                    best_layers = scored[0][1]
                    best_comps = best_comps_candidate
                    best_len = len(best_comps)
        
        # Create next generation
        new_population = []
        
        # Elitism
        for i in range(min(5, len(scored))):
            if scored[i][0] > -10000:
                new_population.append(scored[i][1])
        
        # Crossover and mutation
        while len(new_population) < pop_size:
            if len(new_population) < 2:
                parent1 = best_layers
                parent2 = best_layers
            else:
                candidates = random.sample(scored, min(3, len(scored)))
                candidates.sort(key=lambda x: x[0])
                parent1 = candidates[0][1]
                parent2 = candidates[1][1] if len(candidates) > 1 else best_layers
            
            child = _crossover_layers(parent1, parent2)
            child = _mutate_layers(child, mutation_rate=0.15)
            new_population.append(child)
        
        population = new_population
        
        # Local search on best candidate occasionally
        if generation % 5 == 0 and time.time() - start_time < max_time - 10:
            local_result = _local_search_layers(best_layers, max_iters=20)
            local_comps = _layers_to_comparators(local_result)
            if len(local_comps) < best_len:
                if _evaluate_network_full(local_comps):
                    best_layers = local_result
                    best_comps = local_comps
                    best_len = len(local_comps)
    
    return best_comps

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
