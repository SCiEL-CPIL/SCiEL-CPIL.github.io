"""Layer-based genetic algorithm for sorting networks on 17 wires."""
import random
import numpy as np
import time

N = 17

def _apply_comparator(state, i, j):
    """Apply a compare-exchange to a state vector (in-place)."""
    if state[i] > state[j]:
        state[i], state[j] = state[j], state[i]
    return state

def _evaluate_network_fast(comparators, samples):
    """
    Vectorized evaluation of sorting network on multiple Boolean samples.
    Returns True if all samples are sorted.
    """
    if not comparators:
        return np.all(samples == np.sort(samples, axis=1), axis=1).all()
    
    states = samples.copy()
    for i, j in comparators:
        # Compare-exchange: min to i, max to j
        tmp = np.minimum(states[:, i], states[:, j])
        states[:, i] = tmp
        states[:, j] = np.maximum(states[:, i], states[:, j])
    
    # Check if all rows are sorted
    return np.all(states[:, :-1] <= states[:, 1:], axis=1).all()

def _evaluate_network(comparators, sample_size=2048):
    """
    Evaluate a sorting network using the zero-one principle with random Boolean inputs.
    Returns True if the network sorts all sampled inputs.
    """
    samples = np.random.randint(0, 2, size=(sample_size, N))
    return _evaluate_network_fast(comparators, samples)

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

def _layers_to_comparators(layers):
    """Convert list of layers (each a list of (i,j) pairs) to flat comparator list."""
    comps = []
    for layer in layers:
        for i, j in layer:
            comps.append((i, j))
    return comps

def _comparators_to_layers(comparators):
    """Convert flat comparator list to layers based on dependency."""
    level = [0] * N
    layers = []
    for i, j in comparators:
        lvl = max(level[i], level[j]) + 1
        level[i] = lvl
        level[j] = lvl
        while len(layers) < lvl:
            layers.append([])
        layers[lvl - 1].append((i, j))
    return layers

def _generate_max_matching_layer(used_wires):
    """Generate a random maximum matching layer avoiding used wires."""
    available = [w for w in range(N) if w not in used_wires]
    random.shuffle(available)
    layer = []
    used_in_layer = set()
    for w in available:
        if w in used_in_layer:
            continue
        # Find a partner
        for v in available:
            if v != w and v not in used_in_layer:
                layer.append((min(w, v), max(w, v)))
                used_in_layer.add(w)
                used_in_layer.add(v)
                break
    return layer

def _generate_seed_layers():
    """Generate odd-even transposition network as layers."""
    layers = []
    for phase in range(N):
        layer = []
        for i in range(phase % 2, N - 1, 2):
            layer.append((i, i + 1))
        if layer:
            layers.append(layer)
    return layers

def _mutate_layers(layers, mutation_rate=0.1):
    """
    Apply mutations to layer-based representation.
    Mutations:
    - Swap comparators within a layer
    - Move a comparator to a different layer (if valid)
    - Remove a comparator
    - Insert a new comparator into a layer
    """
    new_layers = [list(layer) for layer in layers]
    total_comps = sum(len(layer) for layer in new_layers)
    num_mutations = max(1, int(mutation_rate * total_comps))
    
    for _ in range(num_mutations):
        op = random.choice(['swap_intra', 'move_inter', 'remove', 'insert'])
        
        if op == 'swap_intra' and len(new_layers) > 0:
            # Swap two comparators within the same layer
            layer_idx = random.randint(0, len(new_layers) - 1)
            if len(new_layers[layer_idx]) >= 2:
                i, j = random.sample(range(len(new_layers[layer_idx])), 2)
                new_layers[layer_idx][i], new_layers[layer_idx][j] = new_layers[layer_idx][j], new_layers[layer_idx][i]
        
        elif op == 'move_inter' and len(new_layers) > 1:
            # Move a comparator from one layer to another
            src_layer = random.randint(0, len(new_layers) - 1)
            if len(new_layers[src_layer]) > 0:
                comp_idx = random.randint(0, len(new_layers[src_layer]) - 1)
                comp = new_layers[src_layer].pop(comp_idx)
                dst_layer = random.randint(0, len(new_layers) - 1)
                # Check if wires are free in destination layer
                wires_used = set()
                for c in new_layers[dst_layer]:
                    wires_used.add(c[0])
                    wires_used.add(c[1])
                if comp[0] not in wires_used and comp[1] not in wires_used:
                    new_layers[dst_layer].append(comp)
                else:
                    new_layers[src_layer].append(comp)
        
        elif op == 'remove' and total_comps > 0:
            # Remove a random comparator
            layer_idx = random.randint(0, len(new_layers) - 1)
            if len(new_layers[layer_idx]) > 0:
                new_layers[layer_idx].pop(random.randint(0, len(new_layers[layer_idx]) - 1))
        
        elif op == 'insert':
            # Insert a new comparator into a random layer
            if len(new_layers) > 0:
                layer_idx = random.randint(0, len(new_layers) - 1)
                wires_used = set()
                for c in new_layers[layer_idx]:
                    wires_used.add(c[0])
                    wires_used.add(c[1])
                available = [w for w in range(N) if w not in wires_used]
                if len(available) >= 2:
                    i, j = random.sample(available, 2)
                    new_layers[layer_idx].append((min(i, j), max(i, j)))
    
    # Remove empty layers
    new_layers = [layer for layer in new_layers if layer]
    return new_layers

def _crossover_layers(parent1, parent2):
    """
    Crossover: take layers from each parent up to a random split point.
    """
    if not parent1:
        return [list(layer) for layer in parent2]
    if not parent2:
        return [list(layer) for layer in parent1]
    
    max_depth = max(len(parent1), len(parent2))
    split = random.randint(0, max_depth)
    
    child = []
    for i in range(max_depth):
        if i < split:
            if i < len(parent1):
                child.append(list(parent1[i]))
            else:
                child.append([])
        else:
            if i < len(parent2):
                child.append(list(parent2[i]))
            else:
                child.append([])
    
    # Remove empty layers
    child = [layer for layer in child if layer]
    return child

def _local_search_layers(layers, max_iters=30, sample_size=512):
    """
    Local search: try removing redundant comparators from layers.
    """
    new_layers = [list(layer) for layer in layers]
    changed = True
    iters = 0
    
    while changed and iters < max_iters:
        changed = False
        iters += 1
        
        # Try removing each comparator
        for layer_idx in range(len(new_layers) - 1, -1, -1):
            for comp_idx in range(len(new_layers[layer_idx]) - 1, -1, -1):
                # Temporarily remove
                comp = new_layers[layer_idx].pop(comp_idx)
                candidate_layers = new_layers[:]
                
                # Quick check
                comps = _layers_to_comparators(candidate_layers)
                if _evaluate_network(comps, sample_size=sample_size):
                    # Keep the removal
                    if not new_layers[layer_idx]:
                        new_layers.pop(layer_idx)
                        # Adjust indices if needed
                        break
                    changed = True
                    break
                else:
                    # Restore
                    new_layers[layer_idx].insert(comp_idx, comp)
            
            if changed:
                break
    
    return new_layers

def construct():
    """
    Construct a sorting network using a layer-based genetic algorithm.
    Returns a list of (i, j) comparator pairs.
    """
    start_time = time.time()
    max_time = 55  # Leave some buffer
    
    # Initialize with seed network as layers
    best_layers = _generate_seed_layers()
    best_comps = _layers_to_comparators(best_layers)
    best_len = len(best_comps)
    
    # Population
    pop_size = 15
    population = [best_layers]
    for _ in range(pop_size - 1):
        mutated = _mutate_layers(best_layers, mutation_rate=0.2)
        population.append(mutated)
    
    # Genetic algorithm loop
    generation = 0
    while time.time() - start_time < max_time:
        generation += 1
        
        # Evaluate and select
        scored = []
        for layers in population:
            comps = _layers_to_comparators(layers)
            # Quick evaluation with random samples
            if _evaluate_network(comps, sample_size=1024):
                score = len(comps)  # Lower is better
            else:
                score = 10000
            scored.append((score, layers))
        
        scored.sort(key=lambda x: x[0])  # Best first (fewest comparators)
        
        # Update best
        if scored[0][0] < 10000:  # Valid
            best_comp_candidate = _layers_to_comparators(scored[0][1])
            if len(best_comp_candidate) < best_len:
                # Full verification
                if _evaluate_network_full(best_comp_candidate):
                    best_layers = scored[0][1]
                    best_comps = best_comp_candidate
                    best_len = len(best_comps)
        
        # Create next generation
        new_population = []
        
        # Elitism: keep best few
        for i in range(min(4, len(scored))):
            if scored[i][0] < 10000:
                new_population.append(scored[i][1])
        
        # Crossover and mutation
        while len(new_population) < pop_size:
            if len(new_population) < 2:
                parent1 = best_layers
                parent2 = best_layers
            else:
                # Tournament selection
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
            local_result = _local_search_layers(best_layers, max_iters=15, sample_size=512)
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
