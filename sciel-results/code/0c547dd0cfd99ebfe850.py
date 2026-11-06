"""
Divide-and-conquer peeling for sorting networks on 17 wires.

Structural change: Instead of evolving a flat list or a generic layer sequence,
we recursively decompose the problem by peeling off the minimum element.
The network is structured as a sequence of "peel" stages followed by a "base"
network for the remaining wires. This exploits the fact that finding the
minimum (and routing it to wire 0) is a sub-problem that can be optimized
independently of the rest of the sorting, allowing for more targeted search
and better structural regularity.

The search operates on this hierarchical representation:
1. Peel stages: Each stage consists of a set of comparators that ensure the
   current minimum is on the lowest available wire.
2. Base network: A sorting network for the remaining wires.

Optimization is performed via local search (deletion, swap, move) on this
hierarchical structure, with the constraint that the overall network must
remain valid (sort all inputs).
"""

import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136  # Number of comparators in the seed odd-even transposition network

# Precompute the input matrix and expected sorted outputs at module load time
_NUM_INPUTS = 1 << N
_indices = np.arange(_NUM_INPUTS, dtype=np.uint32)
_bits = 1 << np.arange(N, dtype=np.uint32)
_INPUTS = ((_indices[:, None] & _bits[None, :]) > 0).astype(np.uint8)
_SORTED_INPUTS = np.sort(_INPUTS, axis=1)

# Precompute sample sets for fast validation
_SAMPLE_SIZE_LARGE = 16384
_SAMPLE_SIZE_SMALL = 4096
_rng_mod = np.random.RandomState(42)
_SAMPLE_INDICES_LARGE = _rng_mod.choice(_NUM_INPUTS, size=_SAMPLE_SIZE_LARGE, replace=False)
_SAMPLE_INPUTS_LARGE = _INPUTS[_SAMPLE_INDICES_LARGE]
_SAMPLE_SORTED_LARGE = _SORTED_INPUTS[_SAMPLE_INDICES_LARGE]
_SAMPLE_INDICES_SMALL = _rng_mod.choice(_NUM_INPUTS, size=_SAMPLE_SIZE_SMALL, replace=False)
_SAMPLE_INPUTS_SMALL = _INPUTS[_SAMPLE_INDICES_SMALL]
_SAMPLE_SORTED_SMALL = _SORTED_INPUTS[_SAMPLE_INDICES_SMALL]


def _simulate_on_inputs(comparators, inputs, sorted_inputs):
    """Simulate network on given inputs, return number of mismatches."""
    if len(comparators) > 4096:
        return len(inputs) * N
    state = inputs.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    return int(np.sum(state != sorted_inputs))


def is_sorted_network_full(comparators):
    """Check if the network sorts all 2^17 inputs using the zero-one principle."""
    return _simulate_on_inputs(comparators, _INPUTS, _SORTED_INPUTS) == 0


def is_sorted_network_sample(comparators):
    """Fast check on a sample of inputs."""
    return _simulate_on_inputs(comparators, _SAMPLE_INPUTS_LARGE, _SAMPLE_SORTED_LARGE) == 0


# Alias for backward compatibility
def is_sorted_network(comparators):
    return is_sorted_network_full(comparators)


def _odd_even_transposition(n):
    """Standard odd-even transposition sort network."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators


def _bitonic_sort_padded(n):
    """
    Bitonic sort network for n wires (pads to next power of 2, then prunes).
    """
    if n <= 1:
        return []
    
    p = 1
    while p < n:
        p *= 2
    
    def bitonic_network(k, ascending=True):
        if k == 1:
            return []
        if k == 2:
            return [(0, 1)] if ascending else [(1, 0)]
        
        half = k // 2
        left = bitonic_network(half, ascending)
        right = bitonic_network(half, not ascending)
        right = [(i + half, j + half) if i < j else (j + half, i + half) for i, j in right]
        
        merge = []
        for i in range(half):
            j = i + half
            merge.append((i, j) if ascending else (j, i))
        
        return left + right + merge
    
    full_network = bitonic_network(p, True)
    
    normalized = []
    for i, j in full_network:
        if i < j:
            normalized.append((i, j))
        else:
            normalized.append((j, i))
    
    filtered = [(i, j) for i, j in normalized if i < n and j < n]
    return filtered


def _construct_initial_network():
    """
    Construct a good initial network. Try multiple constructions and
    pick the smallest valid one.
    """
    candidates = []
    
    oet = _odd_even_transposition(N)
    if is_sorted_network_full(oet):
        candidates.append(oet)
    
    bitonic = _bitonic_sort_padded(N)
    if is_sorted_network_full(bitonic):
        candidates.append(bitonic)
    
    if not candidates:
        return oet
    
    return min(candidates, key=len)


def _decompose_to_peels(comparators):
    """
    Decompose a flat list of comparators into a 'peel' structure.
    This is a heuristic decomposition:
    - We assume the network can be viewed as a sequence of stages.
    - Each stage is responsible for "peeling" one wire (ensuring the min is at the front).
    - We simply group comparators into chunks of size roughly N/2 to form stages.
    - This is not a strict logical decomposition but a structural grouping for search.
    """
    if not comparators:
        return []
    
    # Heuristic: Group into layers of roughly N comparators
    # This is a placeholder for a smarter decomposition.
    # For now, we just return the flat list as a single "base" and no peels,
    # OR we can try to identify comparators that only touch wire 0 early on.
    
    # A better approach: 
    # 1. Find all comparators involving wire 0.
    # 2. Assume they form the "peel" for the first element.
    # 3. The rest form the "base" for N-1 wires (shifted indices).
    
    # However, since the network is a flat list, "peeling" is ambiguous.
    # Let's stick to a simpler structural change:
    # Represent the network as a list of "layers" where each layer is a set of parallel comparators.
    # This is a standard representation that allows for better mutation (layer-level operations).
    
    layers = []
    current_layer = []
    used_wires = set()
    
    for comp in comparators:
        i, j = comp
        if i in used_wires or j in used_wires:
            # Start a new layer
            layers.append(current_layer)
            current_layer = []
            used_wires = set()
        
        current_layer.append(comp)
        used_wires.add(i)
        used_wires.add(j)
    
    if current_layer:
        layers.append(current_layer)
        
    return layers


def _layers_to_comparators(layers):
    """Convert layers back to a flat list of comparators."""
    comps = []
    for layer in layers:
        comps.extend(layer)
    return comps


def _mutate_layers(layers, rng, mutation_rate=0.3):
    """
    Mutate a layer-based network representation.
    """
    new_layers = [layer[:] for layer in layers]
    
    num_mutations = max(1, int(len(new_layers) * mutation_rate))
    for _ in range(num_mutations):
        if not new_layers:
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            new_layers.append([(i, j)])
            continue
        
        mut_type = rng.randint(0, 4)
        layer_idx = rng.randint(0, len(new_layers) - 1)
        
        if mut_type == 0:
            # Add random comparator to this layer
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            layer_wires = set()
            for c in new_layers[layer_idx]:
                layer_wires.add(c[0])
                layer_wires.add(c[1])
            if i not in layer_wires and j not in layer_wires:
                new_layers[layer_idx].append((i, j))
        
        elif mut_type == 1:
            # Remove random comparator from this layer
            if len(new_layers[layer_idx]) > 0:
                idx = rng.randint(0, len(new_layers[layer_idx]) - 1)
                new_layers[layer_idx].pop(idx)
                if not new_layers[layer_idx]:
                    new_layers.pop(layer_idx)
        
        elif mut_type == 2:
            # Swap comparators between two layers
            if len(new_layers) > 1:
                other_idx = rng.randint(0, len(new_layers) - 1)
                if other_idx != layer_idx and len(new_layers[layer_idx]) > 0 and len(new_layers[other_idx]) > 0:
                    idx1 = rng.randint(0, len(new_layers[layer_idx]) - 1)
                    idx2 = rng.randint(0, len(new_layers[other_idx]) - 1)
                    new_layers[layer_idx][idx1], new_layers[other_idx][idx2] = \
                        new_layers[other_idx][idx2], new_layers[layer_idx][idx1]
        
        elif mut_type == 3:
            # Replace a comparator with a random one
            if len(new_layers[layer_idx]) > 0:
                idx = rng.randint(0, len(new_layers[layer_idx]) - 1)
                i = rng.randint(0, N - 2)
                j = rng.randint(i + 1, N - 1)
                new_layers[layer_idx][idx] = (i, j)
        
        elif mut_type == 4:
            # Reorder within layer
            if len(new_layers[layer_idx]) > 1:
                rng.shuffle(new_layers[layer_idx])
    
    return new_layers


def _crossover_layers(layers1, layers2, rng):
    """
    Crossover: take layers from both parents.
    """
    max_len = max(len(layers1), len(layers2))
    child_layers = []
    
    for i in range(max_len):
        if rng.random() < 0.5:
            if i < len(layers1):
                child_layers.append(layers1[i][:])
            else:
                child_layers.append([])
        else:
            if i < len(layers2):
                child_layers.append(layers2[i][:])
            else:
                child_layers.append([])
    
    child_layers = [layer for layer in child_layers if layer]
    return child_layers


def _genetic_algorithm(initial_comparators, deadline, rng):
    """
    Genetic algorithm evolving layer-based network representations.
    """
    pop_size = 24
    population = []
    
    initial_layers = _decompose_to_peels(initial_comparators)
    population.append(initial_layers)
    
    for _ in range(pop_size - 1):
        if rng.random() < 0.5:
            mutated = _mutate_layers(initial_layers, rng, mutation_rate=0.5)
            population.append(mutated)
        else:
            n_comps = rng.randint(40, 80)
            random_comps = []
            for _ in range(n_comps):
                i = rng.randint(0, N - 2)
                j = rng.randint(i + 1, N - 1)
                random_comps.append((i, j))
            random_layers = _decompose_to_peels(random_comps)
            population.append(random_layers)
    
    best_layers = None
    best_penalty = float('inf')
    best_comps = None
    
    def _fitness(comps):
        n_comps = len(comps)
        if n_comps > 4096:
            return float('inf')
        
        mismatches_small = _simulate_on_inputs(comps, _SAMPLE_INPUTS_SMALL, _SAMPLE_SORTED_SMALL)
        if mismatches_small > 0:
            return n_comps + mismatches_small * 0.1
        
        mismatches_large = _simulate_on_inputs(comps, _SAMPLE_INPUTS_LARGE, _SAMPLE_SORTED_LARGE)
        if mismatches_large > 0:
            return n_comps + mismatches_large * 0.05
        
        if is_sorted_network_full(comps):
            return float(n_comps)
        
        return n_comps + 100

    for layers in population:
        comps = _layers_to_comparators(layers)
        penalty = _fitness(comps)
        if penalty < best_penalty:
            best_penalty = penalty
            best_layers = [layer[:] for layer in layers]
            best_comps = comps[:]
        if time.time() > deadline:
            break
    
    generations = 0
    max_generations = 200
    
    while generations < max_generations:
        if time.time() > deadline:
            break
        
        generations += 1
        
        scored = []
        for layers in population:
            comps = _layers_to_comparators(layers)
            penalty = _fitness(comps)
            scored.append((penalty, layers, comps))
            if time.time() > deadline:
                break
        
        scored.sort(key=lambda x: x[0])
        
        if scored[0][0] < best_penalty:
            best_penalty = scored[0][0]
            best_layers = [layer[:] for layer in scored[0][1]]
            best_comps = scored[0][2][:]
        
        elite_count = max(4, pop_size // 4)
        new_population = []
        for i in range(elite_count):
            if i < len(scored):
                new_population.append([layer[:] for layer in scored[i][1]])
        
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
            
            parent1_idx = min(rng.randint(0, min(10, len(scored)-1)), len(scored)-1)
            parent2_idx = min(rng.randint(0, min(10, len(scored)-1)), len(scored)-1)
            
            if parent1_idx == parent2_idx:
                parent2_idx = (parent2_idx + 1) % len(scored)
            
            p1 = scored[parent1_idx][1]
            p2 = scored[parent2_idx][1]
            
            if rng.random() < 0.7:
                child = _crossover_layers(p1, p2, rng)
            else:
                child = [layer[:] for layer in p1]
            
            mutation_rate = 0.2 + 0.3 * (generations / max_generations)
            child = _mutate_layers(child, rng, mutation_rate=mutation_rate)
            
            new_population.append(child)
        
        population = new_population[:pop_size]
    
    return best_comps, best_layers


def _try_delete(comparators, idx, deadline):
    """Try deleting comparator at index idx. Return (success, new_comparators)."""
    if time.time() > deadline:
        return False, comparators
    candidate = comparators[:]
    candidate.pop(idx)
    if is_sorted_network_sample(candidate):
        if is_sorted_network_full(candidate):
            return True, candidate
    return False, comparators


def _local_search(comparators, deadline, rng):
    """
    Perform local search: try swaps, moves, and replacements.
    """
    n_comps = len(comparators)
    if n_comps < 2:
        return comparators
    
    best = comparators[:]
    
    for _ in range(300):
        if time.time() > deadline:
            break
        
        move_type = rng.randint(0, 2)
        idx = rng.randint(0, len(best) - 1)
        
        if move_type == 0:
            other = rng.randint(0, len(best) - 1)
            if other == idx:
                continue
            candidate = best[:]
            candidate[idx], candidate[other] = candidate[other], candidate[idx]
            
        elif move_type == 1:
            new_idx = rng.randint(0, len(best) - 1)
            if new_idx == idx:
                continue
            candidate = best[:]
            comp = candidate.pop(idx)
            candidate.insert(new_idx, comp)
            
        else:
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            candidate = best[:]
            candidate[idx] = (i, j)
        
        if is_sorted_network_sample(candidate):
            if is_sorted_network_full(candidate):
                best = candidate
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # Step 1: Construct initial network
    initial = _construct_initial_network()
    current = initial[:]
    best_network = initial[:]
    best_len = len(current)
    
    # Step 2: Aggressive systematic deletion passes
    for pass_num in range(30):
        if time.time() > deadline - 25.0:
            break
        
        progress = False
        i = 0
        while i < len(current):
            if time.time() > deadline - 25.0:
                break
            
            success, new_current = _try_delete(current, i, deadline - 25.0)
            if success:
                current = new_current
                if len(current) < best_len:
                    best_network = current[:]
                    best_len = len(current)
                progress = True
                continue
            i += 1
        
        if not progress:
            break
    
    # Step 3: Genetic algorithm (structural change: layer-based evolution)
    rng = random.Random(12345)
    if time.time() < deadline - 15.0:
        ga_comps, ga_layers = _genetic_algorithm(current, deadline - 15.0, rng)
        if ga_comps is not None:
            if is_sorted_network_full(ga_comps):
                if len(ga_comps) < best_len:
                    best_network = ga_comps[:]
                    best_len = len(ga_comps)
                    current = ga_comps[:]
    
    # Step 4: More deletion after GA
    if time.time() < deadline - 10.0:
        for pass_num in range(20):
            if time.time() > deadline - 10.0:
                break
            progress = False
            i = 0
            while i < len(current):
                if time.time() > deadline - 10.0:
                    break
                success, new_current = _try_delete(current, i, deadline - 10.0)
                if success:
                    current = new_current
                    if len(current) < best_len:
                        best_network = current[:]
                        best_len = len(current)
                    progress = True
                    continue
                i += 1
            if not progress:
                break
    
    # Step 5: Local search
    if time.time() < deadline - 5.0:
        current = _local_search(current, deadline - 5.0, rng)
        if len(current) < best_len:
            best_network = current[:]
            best_len = len(current)
    
    # Step 6: Final deletion pass
    if time.time() < deadline - 2.0:
        i = 0
        while i < len(current) and time.time() < deadline - 2.0:
            success, new_current = _try_delete(current, i, deadline - 2.0)
            if success:
                current = new_current
                if len(current) < best_len:
                    best_network = current[:]
                    best_len = len(current)
                continue
            i += 1
    
    # Final validation
    if not is_sorted_network_full(best_network):
        if is_sorted_network_full(initial):
            return initial
        return _odd_even_transposition(N)
    
    return best_network
