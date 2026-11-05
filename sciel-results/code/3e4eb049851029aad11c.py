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

def _evaluate_network_samples(comparators, samples):
    """
    Evaluate a sorting network using the zero-one principle with random Boolean inputs.
    Returns True if the network sorts all sampled inputs.
    """
    for sample in samples:
        state = sample.copy()
        for i, j in comparators:
            if state[i] > state[j]:
                state[i], state[j] = state[j], state[i]
        if not np.all(state[:-1] <= state[1:]):
            return False
    return True

def _flatten_layers(layers):
    """Flatten a list of layers into a single list of comparators."""
    flat = []
    for layer in layers:
        flat.extend(layer)
    return flat

def _flat_to_layers(flat_comparators):
    """
    Convert a flat list of comparators into a sequence of parallel layers.
    Uses a greedy algorithm: assign each comparator to the earliest possible layer
    where its wires are not already used.
    """
    layers = []
    for comp in flat_comparators:
        i, j = comp
        placed = False
        for k in range(len(layers)):
            if i not in layers[k] and j not in layers[k]:
                layers[k].append(comp)
                placed = True
                break
        if not placed:
            layers.append([comp])
    return layers

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
    # Assign each comparator to a layer (level)
    # A comparator can be at level max(level[i], level[j]) + 1
    level = [0] * N
    depth = 0
    for i, j in comparators:
        lvl = max(level[i], level[j]) + 1
        level[i] = lvl
        level[j] = lvl
        depth = max(depth, lvl)
    return depth

def _generate_seed_network():
    """Generate the odd-even transposition network as seed."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def _mutate_layers(layers, mutation_rate=0.1):
    """
    Mutate a network represented as layers.
    1. Remove a comparator from a random layer.
    2. Insert a random comparator into the network (re-layering if needed).
    3. Swap two comparators between layers (if valid).
    """
    new_layers = [list(layer) for layer in layers]
    n_comps = sum(len(l) for l in new_layers)
    
    num_mutations = max(1, int(mutation_rate * n_comps))
    
    for _ in range(num_mutations):
        op = random.choice(['remove', 'insert', 'swap_pos'])
        
        if op == 'remove' and n_comps > 0:
            flat_idx = random.randint(0, n_comps - 1)
            count = 0
            for k in range(len(new_layers)):
                if count + len(new_layers[k]) > flat_idx:
                    local_idx = flat_idx - count
                    new_layers[k].pop(local_idx)
                    n_comps -= 1
                    break
                count += len(new_layers[k])
            new_layers = [l for l in new_layers if l]
            
        elif op == 'insert':
            i, j = random.sample(range(N), 2)
            if i > j: i, j = j, i
            new_comp = (i, j)
            inserted = False
            for k in range(len(new_layers)):
                if i not in new_layers[k] and j not in new_layers[k]:
                    new_layers[k].append(new_comp)
                    inserted = True
                    n_comps += 1
                    break
            if not inserted:
                new_layers.append([new_comp])
                n_comps += 1
                
        elif op == 'swap_pos' and n_comps > 1:
            flat = _flatten_layers(new_layers)
            idx1, idx2 = random.sample(range(len(flat)), 2)
            flat[idx1], flat[idx2] = flat[idx2], flat[idx1]
            new_layers = _flat_to_layers(flat)
            
    return new_layers

def _crossover_layers(parent1, parent2):
    """
    Crossover: Flatten both, uniform crossover, then re-layer.
    """
    flat1 = _flatten_layers(parent1)
    flat2 = _flatten_layers(parent2)
    
    child_flat = []
    for i in range(max(len(flat1), len(flat2))):
        if i < len(flat1) and i < len(flat2):
            child_flat.append(flat1[i] if random.random() < 0.5 else flat2[i])
        elif i < len(flat1):
            child_flat.append(flat1[i])
        else:
            child_flat.append(flat2[i])
            
    return _flat_to_layers(child_flat)

def _local_search_layers(layers, samples, max_iters=20):
    """
    Local search: try removing comparators that are redundant.
    """
    current_layers = [list(l) for l in layers]
    changed = True
    iters = 0
    while changed and iters < max_iters:
        changed = False
        iters += 1
        flat = _flatten_layers(current_layers)
        for idx in range(len(flat) - 1, -1, -1):
            candidate_flat = flat[:idx] + flat[idx+1:]
            if _evaluate_network_samples(candidate_flat, samples):
                current_layers = _flat_to_layers(candidate_flat)
                changed = True
                break
    return current_layers

def construct():
    """
    Construct a sorting network using a genetic algorithm approach with layer representation.
    Returns a list of (i, j) comparator pairs.
    """
    start_time = time.time()
    max_time = 55  # Leave some buffer
    
    # Initialize with seed network (converted to layers)
    seed_flat = _generate_seed_network()
    seed_layers = _flat_to_layers(seed_flat)
    
    best_flat = seed_flat
    best_len = len(best_flat)
    
    # Population of layers
    pop_size = 30
    population = [seed_layers]
    for _ in range(pop_size - 1):
        mutated = _mutate_layers(seed_layers, mutation_rate=0.3)
        population.append(mutated)
    
    # Pre-generate samples for fast evaluation
    samples = np.random.randint(0, 2, size=(1024, N))
    
    generation = 0
    while time.time() - start_time < max_time:
        generation += 1
        
        scored = []
        for layers in population:
            flat = _flatten_layers(layers)
            # Quick check
            if _evaluate_network_samples(flat, samples):
                score = len(flat)
            else:
                score = float('inf')
            scored.append((score, layers))
        
        scored.sort(key=lambda x: x[0])
        
        # Update best if we find something shorter and it passes full check
        if scored[0][0] < best_len:
            candidate_flat = _flatten_layers(scored[0][1])
            if _evaluate_network_full(candidate_flat):
                best_flat = candidate_flat
                best_len = len(candidate_flat)
        
        # Create next generation
        new_population = []
        
        # Elitism
        for i in range(min(5, len(scored))):
            if scored[i][0] < float('inf'):
                new_population.append(scored[i][1])
        
        # Fill rest with offspring
        while len(new_population) < pop_size:
            if len(new_population) < 2:
                p1 = scored[0][1]
                p2 = scored[0][1]
            else:
                # Tournament selection
                c1 = random.sample(scored, 3)
                c1.sort(key=lambda x: x[0])
                p1 = c1[0][1]
                c2 = random.sample(scored, 3)
                c2.sort(key=lambda x: x[0])
                p2 = c2[0][1]
                
            child = _crossover_layers(p1, p2)
            child = _mutate_layers(child, mutation_rate=0.1)
            new_population.append(child)
            
        population = new_population
        
        # Local Search on the best candidate occasionally
        if generation % 10 == 0 and time.time() - start_time < max_time - 5:
            best_candidate = scored[0][1]
            local_best = _local_search_layers(best_candidate, samples, max_iters=20)
            local_flat = _flatten_layers(local_best)
            if len(local_flat) < best_len:
                if _evaluate_network_full(local_flat):
                    best_flat = local_flat
                    best_len = len(local_flat)
    
    return best_flat

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
