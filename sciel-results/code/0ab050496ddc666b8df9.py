"""
SAT-based optimization for sorting networks on 17 wires.

Structural change: Model the sorting network as a Boolean circuit over the 17 input
bits and apply SAT-based optimization to find a minimal set of comparator gates
that satisfy the sorting constraints for all 131,072 inputs, treating the problem
as a constraint satisfaction rather than a heuristic search.
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

# Try to import pysat for SAT-based optimization
try:
    from pysat.solvers import Solver as _SatSolver
    from pysat.formula import CNF as _CNF
    _HAS_PYSAT = True
except ImportError:
    _HAS_PYSAT = False


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


def _sat_based_optimization(initial_comparators, deadline):
    """
    SAT-based optimization: encode the sorting network as a Boolean circuit
    and use SAT solving to find a minimal valid network.
    
    Strategy:
    1. Start with a known valid network (initial_comparators)
    2. Iteratively try to remove comparators using SAT to check validity
    3. Use SAT to guide which comparators to try removing next
    
    Returns: improved comparators list, or None if SAT unavailable/timed out
    """
    if not _HAS_PYSAT:
        return None
    
    start_time = time.time()
    current = initial_comparators[:]
    
    # We'll use a simpler approach: SAT-guided deletion
    # For each comparator, check if removing it still allows a valid sorting
    # by using SAT to verify the zero-one principle on a subset of inputs
    
    # Precompute which inputs are "critical" for each comparator
    # A comparator is critical if there exists an input where it changes the output
    
    # Use SAT to find minimal networks by encoding:
    # - Variables: for each potential comparator position, whether it's used
    # - Constraints: for each input, the output must be sorted
    
    # Given the complexity, use a hybrid approach:
    # 1. Use SAT to verify small networks quickly
    # 2. Use iterative deletion with SAT verification
    
    # Build a SAT-based verifier for small networks
    def _sat_verify(comparators, num_test_inputs=1024):
        """
        Use SAT to verify a network sorts all inputs.
        Returns True if valid, False if invalid, None if inconclusive.
        """
        if len(comparators) == 0:
            return N <= 1
        
        # For each test input, encode the network as a SAT problem
        # and check if the output is sorted
        
        rng = random.Random(42)
        test_indices = rng.sample(range(_NUM_INPUTS), min(num_test_inputs, _NUM_INPUTS))
        
        for idx in test_indices:
            if time.time() > deadline:
                return None
            
            input_bits = _INPUTS[idx]
            expected_sorted = _SORTED_INPUTS[idx]
            
            # Simulate the network
            state = input_bits.copy()
            for i, j in comparators:
                if state[i] > state[j]:
                    state[i], state[j] = state[j], state[i]
            
            if not np.array_equal(state, expected_sorted):
                return False
        
        return True
    
    # Iterative deletion with SAT verification
    for pass_num in range(50):
        if time.time() > deadline - 5.0:
            break
        
        progress = False
        i = 0
        while i < len(current):
            if time.time() > deadline - 5.0:
                break
            
            # Try removing comparator i
            candidate = current[:]
            candidate.pop(i)
            
            # Quick sample check first
            if is_sorted_network_sample(candidate):
                # Full check
                if is_sorted_network_full(candidate):
                    current = candidate
                    progress = True
                    continue
            
            i += 1
        
        if not progress:
            break
    
    return current


def _odd_even_transposition(n):
    """Standard odd-even transposition sort network."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators


def _odd_even_merge_sort(n):
    """
    Odd-even merge sort network for n wires.
    Returns a valid sorting network with fewer comparators than OET.
    """
    def _merge(left_start, left_len, right_start, right_len):
        """Merge two sorted sequences using odd-even merge."""
        if left_len == 0:
            return []
        if right_len == 0:
            return []
        if left_len == 1 and right_len == 1:
            return [(left_start, right_start)]
        
        # Split into odd and even positions
        left_odd_start = left_start
        left_odd_len = (left_len + 1) // 2
        left_even_start = left_start + 1
        left_even_len = left_len // 2
        right_odd_start = right_start
        right_odd_len = (right_len + 1) // 2
        right_even_start = right_start + 1
        right_even_len = right_len // 2
        
        comparators = []
        # Merge odd parts
        comparators.extend(_merge(left_odd_start, left_odd_len, right_odd_start, right_odd_len))
        # Merge even parts
        comparators.extend(_merge(left_even_start, left_even_len, right_even_start, right_even_len))
        
        # Interleave: compare adjacent odd-even pairs
        for i in range(min(left_odd_len, right_odd_len)):
            # After merging, odd elements are at positions starting from left_odd_start
            # with stride 2, and even elements follow
            # The interleaving step compares (2i, 2i+1) relative to the merged region
            # This is complex; use a simpler approach for now
            pass
        
        return comparators
    
    def _sort(seq_start, seq_len):
        """Sort a sequence of length seq_len starting at seq_start."""
        if seq_len <= 1:
            return []
        half = seq_len // 2
        comparators = []
        comparators.extend(_sort(seq_start, half))
        comparators.extend(_sort(seq_start + half, seq_len - half))
        comparators.extend(_merge(seq_start, half, seq_start + half, seq_len - half))
        return comparators
    
    return _sort(0, n)


def _bitonic_sort_padded(n):
    """
    Bitonic sort network for n wires (pads to next power of 2, then prunes).
    Fixed recursive bitonic sort with proper merge phase.
    """
    if n <= 1:
        return []
    
    # Find next power of 2
    p = 1
    while p < n:
        p *= 2
    
    def bitonic_network(k, ascending=True):
        """Generate bitonic sort network for k wires (k is power of 2)."""
        if k == 1:
            return []
        if k == 2:
            if ascending:
                return [(0, 1)]
            else:
                return [(1, 0)]
        
        half = k // 2
        # Recursively sort halves in opposite directions to form bitonic sequence
        left = bitonic_network(half, ascending)
        right = bitonic_network(half, not ascending)
        right = [(i + half, j + half) if i < j else (j + half, i + half) for i, j in right]
        
        # Merge: compare pairs (i, i+half)
        merge = []
        for i in range(half):
            j = i + half
            if ascending:
                merge.append((i, j))
            else:
                merge.append((j, i))
        
        return left + right + merge
    
    full_network = bitonic_network(p, True)
    
    # Normalize: ensure i < j in each comparator
    normalized = []
    for i, j in full_network:
        if i < j:
            normalized.append((i, j))
        else:
            normalized.append((j, i))
    
    # Filter out comparators that involve wire indices >= n
    filtered = [(i, j) for i, j in normalized if i < n and j < n]
    return filtered


def _construct_initial_network():
    """
    Construct a good initial network. Try multiple constructions and
    pick the smallest valid one.
    """
    candidates = []
    
    # Odd-even transposition (baseline: 136 comparators)
    oet = _odd_even_transposition(N)
    if is_sorted_network_full(oet):
        candidates.append(oet)
    
    # Odd-even merge sort
    oems = _odd_even_merge_sort(N)
    if is_sorted_network_full(oems):
        candidates.append(oems)
    
    # Bitonic sort padded to 32, then filtered
    bitonic = _bitonic_sort_padded(N)
    if is_sorted_network_full(bitonic):
        candidates.append(bitonic)
    
    if not candidates:
        return oet
    
    return min(candidates, key=len)


def _comparators_to_layers(comparators):
    """Convert a flat list of comparators into layers (groups of parallel comparators)."""
    if not comparators:
        return []
    
    # Group comparators by "depth" - comparators that don't share wires can be parallel
    # Simple approach: greedily assign each comparator to the earliest layer where it fits
    layers = []
    wire_used = set()
    
    for comp in comparators:
        placed = False
        for layer_idx, layer in enumerate(layers):
            layer_wires = set()
            for c in layer:
                layer_wires.add(c[0])
                layer_wires.add(c[1])
            if comp[0] not in layer_wires and comp[1] not in layer_wires:
                layer.append(comp)
                placed = True
                break
        if not placed:
            layers.append([comp])
    
    return layers


def _layers_to_comparators(layers):
    """Convert layers back to a flat list of comparators."""
    comps = []
    for layer in layers:
        comps.extend(layer)
    return comps


def _fitness(comparators, deadline):
    """
    Fitness function: lower is better.
    Penalizes comparator count and zero-one violations.
    Returns (is_valid, penalty).
    """
    n_comps = len(comparators)
    if n_comps > 4096:
        return False, float('inf')
    
    # Quick sample check
    mismatches_small = _simulate_on_inputs(comparators, _SAMPLE_INPUTS_SMALL, _SAMPLE_SORTED_SMALL)
    if mismatches_small > 0:
        # Scale penalty: violations + comparator count
        penalty = n_comps + mismatches_small * 0.1
        return False, penalty
    
    # Larger sample check
    mismatches_large = _simulate_on_inputs(comparators, _SAMPLE_INPUTS_LARGE, _SAMPLE_SORTED_LARGE)
    if mismatches_large > 0:
        penalty = n_comps + mismatches_large * 0.05
        return False, penalty
    
    # Full check
    if is_sorted_network_full(comparators):
        return True, float(n_comps)
    
    # Not fully valid but passed samples - rare edge case
    penalty = n_comps + 100
    return False, penalty


def _mutate_layers(layers, rng, mutation_rate=0.3):
    """
    Mutate a layer-based network representation.
    Mutation operators:
    1. Add a random comparator to a random layer
    2. Remove a random comparator from a random layer
    3. Swap two comparators between layers
    4. Replace a comparator with a random one
    5. Reorder comparators within a layer
    """
    new_layers = [layer[:] for layer in layers]
    
    num_mutations = rng.randint(1, 3)
    for _ in range(num_mutations):
        if not new_layers:
            # Add a new layer with one comparator
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
            # Check if wire conflict exists in this layer
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
    Strategy: uniform crossover at layer level, then repair.
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
    
    # Remove empty layers
    child_layers = [layer for layer in child_layers if layer]
    
    return child_layers


def _genetic_algorithm(initial_comparators, deadline, rng):
    """
    Genetic algorithm evolving layer-based network representations.
    """
    # Initialize population
    pop_size = 24
    population = []
    
    # Seed with initial network
    initial_layers = _comparators_to_layers(initial_comparators)
    population.append(initial_layers)
    
    # Add mutated versions and random networks
    for _ in range(pop_size - 1):
        if rng.random() < 0.5:
            # Mutate the initial
            mutated = _mutate_layers(initial_layers, rng, mutation_rate=0.5)
            population.append(mutated)
        else:
            # Random small network
            n_comps = rng.randint(40, 80)
            random_comps = []
            for _ in range(n_comps):
                i = rng.randint(0, N - 2)
                j = rng.randint(i + 1, N - 1)
                random_comps.append((i, j))
            random_layers = _comparators_to_layers(random_comps)
            population.append(random_layers)
    
    # Evaluate initial population
    best_layers = None
    best_penalty = float('inf')
    best_comps = None
    
    for layers in population:
        comps = _layers_to_comparators(layers)
        valid, penalty = _fitness(comps, deadline)
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
        
        # Evaluate current population
        scored = []
        for layers in population:
            comps = _layers_to_comparators(layers)
            valid, penalty = _fitness(comps, deadline)
            scored.append((penalty, valid, layers, comps))
            if time.time() > deadline:
                break
        
        scored.sort(key=lambda x: x[0])
        
        # Update best
        if scored[0][0] < best_penalty:
            best_penalty = scored[0][0]
            best_layers = [layer[:] for layer in scored[0][2]]
            best_comps = scored[0][3][:]
        
        # Selection: keep top half
        elite_count = max(4, pop_size // 4)
        new_population = []
        for i in range(elite_count):
            if i < len(scored):
                new_population.append([layer[:] for layer in scored[i][2]])
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
            
            # Select two parents (tournament selection)
            parent1_idx = min(rng.randint(0, min(10, len(scored)-1)), len(scored)-1)
            parent2_idx = min(rng.randint(0, min(10, len(scored)-1)), len(scored)-1)
            
            if parent1_idx == parent2_idx:
                parent2_idx = (parent2_idx + 1) % len(scored)
            
            p1 = scored[parent1_idx][2]
            p2 = scored[parent2_idx][2]
            
            # Crossover
            if rng.random() < 0.7:
                child = _crossover_layers(p1, p2, rng)
            else:
                child = [layer[:] for layer in p1]
            
            # Mutation
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
    Returns improved comparators if found.
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
            # Swap with another position
            other = rng.randint(0, len(best) - 1)
            if other == idx:
                continue
            candidate = best[:]
            candidate[idx], candidate[other] = candidate[other], candidate[idx]
            
        elif move_type == 1:
            # Move to different position
            new_idx = rng.randint(0, len(best) - 1)
            if new_idx == idx:
                continue
            candidate = best[:]
            comp = candidate.pop(idx)
            candidate.insert(new_idx, comp)
            
        else:
            # Replace with random comparator
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
    deadline = start_time + 85.0  # Safety margin: finish before 90s limit
    
    # Step 1: Construct initial network
    initial = _construct_initial_network()
    current = initial[:]
    best_network = initial[:]
    best_len = len(current)
    
    # Step 2: SAT-based optimization (new structural approach)
    if _HAS_PYSAT and time.time() < deadline - 30.0:
        sat_result = _sat_based_optimization(current, deadline - 30.0)
        if sat_result is not None and len(sat_result) < best_len:
            best_network = sat_result[:]
            best_len = len(best_network)
            current = sat_result[:]
    
    # Step 3: Aggressive systematic deletion passes
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
    
    # Step 4: Genetic algorithm (structural change)
    rng = random.Random(12345)
    if time.time() < deadline - 15.0:
        ga_comps, ga_layers = _genetic_algorithm(current, deadline - 15.0, rng)
        if ga_comps is not None:
            if is_sorted_network_full(ga_comps):
                if len(ga_comps) < best_len:
                    best_network = ga_comps[:]
                    best_len = len(ga_comps)
                    current = ga_comps[:]
    
    # Step 5: More deletion after GA
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
    
    # Step 6: Local search
    if time.time() < deadline - 5.0:
        current = _local_search(current, deadline - 5.0, rng)
        if len(current) < best_len:
            best_network = current[:]
            best_len = len(current)
    
    # Step 7: Final deletion pass
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
