"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 17
SEED_COMPARATORS = 136

# Precompute all Boolean inputs
def _make_all_inputs():
    n_inputs = 1 << N
    indices = np.arange(n_inputs, dtype=np.uint32)[:, None]
    bit_positions = np.arange(N, dtype=np.uint32)[None, :]
    inputs = ((indices >> bit_positions) & 1).astype(np.uint8)
    return inputs

_ALL_INPUTS = None
_SAMPLE_INPUTS = None

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        _ALL_INPUTS = _make_all_inputs()
    return _ALL_INPUTS

def _get_sample_inputs(n_samples=2048):
    global _SAMPLE_INPUTS
    if _SAMPLE_INPUTS is None:
        all_inputs = _get_all_inputs()
        rng = np.random.default_rng(12345)
        idx = rng.choice(len(all_inputs), size=n_samples, replace=False)
        _SAMPLE_INPUTS = all_inputs[idx]
    return _SAMPLE_INPUTS

def simulate_network(comparators, inputs):
    """Simulate compare-exchange network. Returns output array."""
    state = inputs.copy()
    for (i, j) in comparators:
        mi = state[:, i] & state[:, j]
        mj = state[:, i] | state[:, j]
        state[:, i] = mi
        state[:, j] = mj
    return state

def _check_sorted(output):
    """Check if output rows are sorted (non-decreasing)."""
    for k in range(N - 1):
        if np.any(output[:, k] > output[:, k + 1]):
            return False
    return True

def is_sorted_fast(comparators, inputs=None):
    """Quick check on sample inputs."""
    if inputs is None:
        inputs = _get_sample_inputs()
    output = simulate_network(comparators, inputs)
    return _check_sorted(output)

def is_sorted_network(comparators):
    """Full check on all 2^N Boolean inputs."""
    inputs = _get_all_inputs()
    output = simulate_network(comparators, inputs)
    return _check_sorted(output)

def odd_even_merge_sort_network(n):
    """
    Generate an odd-even merge sort network for n wires.
    Uses the recursive odd-even merge construction.
    Returns list of (i, j) comparator pairs.
    """
    if n == 1:
        return []
    
    def _build(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        left = _build(half)
        right = [(i + half, j + half) for (i, j) in _build(n - half)]
        merged = _odd_even_merge(n)
        return left + right + merged
    
    def _odd_even_merge(n):
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        comparators = []
        
        if n % 2 == 0:
            for i in range(0, n, 2):
                comparators.append((i, i + 1))
            even_merged = _odd_even_merge(n // 2)
            for (a, b) in even_merged:
                comparators.append((2 * a, 2 * b))
            odd_merged = _odd_even_merge(n // 2)
            for (a, b) in odd_merged:
                comparators.append((2 * a + 1, 2 * b + 1))
        else:
            for i in range(1, n - 1, 2):
                comparators.append((i, i + 1))
            if n - 2 > 0:
                mid_merged = _odd_even_merge(n - 2)
                for (a, b) in mid_merged:
                    comparators.append((a + 1, b + 1))
        
        return comparators
    
    return _build(n)

def build_layers(comparators):
    """
    Build depth-parallel layers from a sequential comparator list.
    Each layer contains disjoint comparators (no shared wires).
    """
    layers = []
    current_layer = []
    used_wires = set()
    
    for (i, j) in comparators:
        if i in used_wires or j in used_wires:
            if current_layer:
                layers.append(current_layer)
            current_layer = []
            used_wires = set()
        current_layer.append((i, j))
        used_wires.add(i)
        used_wires.add(j)
    
    if current_layer:
        layers.append(current_layer)
    
    return layers

def flatten_layers(layers):
    """Flatten layers back to a sequential list."""
    result = []
    for layer in layers:
        result.extend(layer)
    return result

def simulate_layers(layers, inputs):
    """Simulate a network given as layers of disjoint comparators."""
    state = inputs.copy()
    for layer in layers:
        if not layer:
            continue
        # Apply all comparators in this layer
        # Since wires are disjoint, we can do this in parallel or simple loop
        for (i, j) in layer:
            mi = state[:, i] & state[:, j]
            mj = state[:, i] | state[:, j]
            state[:, i] = mi
            state[:, j] = mj
    return state

def is_sorted_network_layers(layers, inputs):
    """Check if a layer-based network sorts given inputs."""
    output = simulate_layers(layers, inputs)
    return _check_sorted(output)

def optimize_layer_structure(layers, deadline):
    """
    Optimize a layer-based network by removing redundant comparators.
    """
    best_layers = [list(layer) for layer in layers]
    
    # Use a small sample for fast pre-filtering
    pre_sample = _get_sample_inputs(1024)
    
    def quick_check(candidate_layers):
        state = pre_sample.copy()
        for layer in candidate_layers:
            if not layer:
                continue
            for (i, j) in layer:
                mi = state[:, i] & state[:, j]
                mj = state[:, i] | state[:, j]
                state[:, i] = mi
                state[:, j] = mj
        for k in range(N - 1):
            if np.any(state[:, k] > state[:, k + 1]):
                return False
        return True
    
    def full_check(candidate_layers):
        return is_sorted_network_layers(candidate_layers, _get_all_inputs())
    
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        
        # 1. Try removing comparators from each layer, starting from the end
        for layer_idx in range(len(best_layers) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            if not best_layers[layer_idx]:
                continue
            
            for comp_idx in range(len(best_layers[layer_idx]) - 1, -1, -1):
                if time.time() > deadline - 5:
                    break
                
                # Create candidate with this comparator removed
                candidate_layers = [list(l) for l in best_layers]
                candidate_layers[layer_idx] = candidate_layers[layer_idx][:comp_idx] + \
                                                 candidate_layers[layer_idx][comp_idx+1:]
                
                # Quick check first
                if quick_check(candidate_layers):
                    # Full check
                    if full_check(candidate_layers):
                        best_layers = candidate_layers
                        improved = True
                        # Continue trying to remove more from this layer (index shifted)
                        if comp_idx > 0:
                            comp_idx -= 1
                    else:
                        comp_idx -= 1
                else:
                    comp_idx -= 1
        
        # 2. Try merging adjacent layers that have no conflicts
        if time.time() < deadline - 10:
            merged = False
            for i in range(len(best_layers) - 1):
                if time.time() > deadline - 5:
                    break
                if not best_layers[i] or not best_layers[i+1]:
                    continue
                
                wires_i = set()
                for (a, b) in best_layers[i]:
                    wires_i.add(a)
                    wires_i.add(b)
                wires_j = set()
                for (a, b) in best_layers[i+1]:
                    wires_j.add(a)
                    wires_j.add(b)
                
                if not wires_i & wires_j:
                    candidate_layers = [list(l) for l in best_layers]
                    candidate_layers[i] = best_layers[i] + best_layers[i+1]
                    candidate_layers.pop(i+1)
                    
                    if quick_check(candidate_layers):
                        if full_check(candidate_layers):
                            best_layers = candidate_layers
                            merged = True
                            break
            
            if merged:
                improved = True
    
    return best_layers

def improve_by_removal_sequential(comparators, deadline):
    """
    Sequential pass: try removing each comparator in order.
    """
    best = list(comparators)
    pre_sample = _get_sample_inputs(2048)
    
    def quick_check(candidate):
        state = pre_sample.copy()
        for (i, j) in candidate:
            mi = state[:, i] & state[:, j]
            mj = state[:, i] | state[:, j]
            state[:, i] = mi
            state[:, j] = mj
        for k in range(N - 1):
            if np.any(state[:, k] > state[:, k + 1]):
                return False
        return True
    
    while time.time() < deadline - 5:
        removed_any = False
        i = 0
        while i < len(best):
            if time.time() > deadline - 5:
                break
            candidate = best[:i] + best[i+1:]
            if quick_check(candidate):
                if is_sorted_network(candidate):
                    best = candidate
                    removed_any = True
                else:
                    i += 1
            else:
                i += 1
        
        if not removed_any:
            break
    
    return best

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Use odd-even merge sort as the base network (109 comparators)
    base = odd_even_merge_sort_network(N)
    
    # Verify base is correct
    if not is_sorted_network(base):
        # Fallback to odd-even transposition
        base = [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    best = list(base)
    
    # Phase 1: Convert to layers and optimize structure
    if time.time() < deadline - 20:
        layers = build_layers(best)
        optimized_layers = optimize_layer_structure(layers, deadline)
        best = flatten_layers(optimized_layers)
    
    # Phase 2: Sequential removal to catch remaining redundancies
    if time.time() < deadline - 10:
        best = improve_by_removal_sequential(best, deadline)
    
    # Phase 3: Convert back to layers and optimize again (structure might have changed)
    if time.time() < deadline - 5:
        layers = build_layers(best)
        optimized_layers = optimize_layer_structure(layers, deadline)
        best = flatten_layers(optimized_layers)
    
    # Phase 4: Final sequential pass
    if time.time() < deadline - 3:
        best = improve_by_removal_sequential(best, deadline)
    
    # Verify final result
    if not is_sorted_network(best):
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    return best
# EVOLVE-BLOCK-END
