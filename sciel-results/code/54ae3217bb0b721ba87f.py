"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import defaultdict

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
_SORTED_INPUTS = None
_SAMPLE_INPUTS = None
_SAMPLE_SORTED = None

def _get_all_inputs():
    global _ALL_INPUTS, _SORTED_INPUTS
    if _ALL_INPUTS is None:
        _ALL_INPUTS = _make_all_inputs()
        # Sort each row to get the expected output
        # np.sort on uint8 array along axis 1
        _SORTED_INPUTS = np.sort(_ALL_INPUTS, axis=1)
    return _ALL_INPUTS, _SORTED_INPUTS

def _get_sample_inputs(n_samples=4096):
    global _SAMPLE_INPUTS, _SAMPLE_SORTED
    if _SAMPLE_INPUTS is None:
        all_inputs, sorted_inputs = _get_all_inputs()
        rng = np.random.default_rng(12345)
        idx = rng.choice(len(all_inputs), size=n_samples, replace=False)
        _SAMPLE_INPUTS = all_inputs[idx]
        _SAMPLE_SORTED = sorted_inputs[idx]
    return _SAMPLE_INPUTS, _SAMPLE_SORTED

def simulate_network(comparators, inputs):
    """Simulate compare-exchange network. Returns output array."""
    state = inputs.copy()
    for (i, j) in comparators:
        a = state[:, i]
        b = state[:, j]
        # swap_mask is 1 where a > b (i.e., a=1, b=0)
        swap_mask = a & ~b
        # Where swap_mask=1: new a = b (0), new b = a (1)
        # Where swap_mask=0: unchanged
        state[:, i] = a & ~swap_mask  # a stays 1 only if not swapped
        state[:, j] = b | swap_mask   # b becomes 1 where swapped
    return state

def is_sorted_network(comparators):
    """Full check on all 2^N Boolean inputs."""
    if len(comparators) > 4096:
        return False
    inputs, sorted_inputs = _get_all_inputs()
    state = inputs.copy()
    for (i, j) in comparators:
        a = state[:, i]
        b = state[:, j]
        swap_mask = a & ~b
        state[:, i] = a & ~swap_mask
        state[:, j] = b | swap_mask
    
    # Check if state matches sorted inputs exactly
    return np.array_equal(state, sorted_inputs)

def is_sorted_fast(comparators, inputs=None, sorted_outputs=None):
    """Quick check on sample inputs."""
    if inputs is None:
        inputs, sorted_outputs = _get_sample_inputs()
    if len(comparators) > 4096:
        return False
    state = inputs.copy()
    for (i, j) in comparators:
        a = state[:, i]
        b = state[:, j]
        swap_mask = a & ~b
        state[:, i] = a & ~swap_mask
        state[:, j] = b | swap_mask
    return np.array_equal(state, sorted_outputs)

def odd_even_merge_sort_network(n):
    """
    Generate an odd-even merge sort network for n wires.
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

def list_to_layers(comparators):
    """
    Convert a flat list of comparators into a list of layers.
    A layer is a list of disjoint comparators.
    """
    layers = []
    current_layer = []
    used_wires = set()
    
    for comp in comparators:
        i, j = comp
        if i in used_wires or j in used_wires:
            if current_layer:
                layers.append(current_layer)
            current_layer = []
            used_wires = set()
        current_layer.append(comp)
        used_wires.add(i)
        used_wires.add(j)
    
    if current_layer:
        layers.append(current_layer)
        
    return layers

def layers_to_list(layers):
    """
    Convert a list of layers back to a flat list of comparators.
    """
    result = []
    for layer in layers:
        result.extend(layer)
    return result

def can_add_to_layer(comp, layer):
    """Check if a comparator can be added to a layer without conflict."""
    i, j = comp
    for (a, b) in layer:
        if i in (a, b) or j in (a, b):
            return False
    return True

def improve_by_removal_sequential(comparators, deadline):
    """
    Sequential pass: try removing each comparator in order.
    """
    best = list(comparators)
    sample_inputs, sample_sorted = _get_sample_inputs(2048)
    
    def quick_check(candidate):
        state = sample_inputs.copy()
        for (i, j) in candidate:
            a = state[:, i]
            b = state[:, j]
            sm = a & ~b
            state[:, i] = a & ~sm
            state[:, j] = b | sm
        return np.array_equal(state, sample_sorted)
    
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

def local_search_layers(initial_comparators, deadline):
    """
    Local search operating on the layer representation.
    Moves:
    1. Remove comparator (if valid).
    2. Move comparator to adjacent layer (if disjoint and valid).
    3. Insert comparator (if it allows removal of 2 others).
    """
    best_comparators = list(initial_comparators)
    best_layers = list_to_layers(best_comparators)
    sample_inputs, sample_sorted = _get_sample_inputs(2048)
    
    def quick_check(comps):
        state = sample_inputs.copy()
        for (i, j) in comps:
            a = state[:, i]
            b = state[:, j]
            sm = a & ~b
            state[:, i] = a & ~sm
            state[:, j] = b | sm
        return np.array_equal(state, sample_sorted)
    
    def full_check(comps):
        return is_sorted_network(comps)
    
    improved = True
    while improved and time.time() < deadline - 10:
        improved = False
        
        # Phase 1: Greedy Removal
        # Flatten and try removing each
        current_list = layers_to_list(best_layers)
        i = 0
        while i < len(current_list):
            if time.time() > deadline - 10:
                break
            candidate = current_list[:i] + current_list[i+1:]
            if quick_check(candidate):
                if full_check(candidate):
                    current_list = candidate
                    best_layers = list_to_layers(current_list)
                    improved = True
                    # Restart scan from 0 as indices changed
                    i = 0
                else:
                    i += 1
            else:
                i += 1
        
        # Phase 2: Layer Shifting
        # Try moving comparators to adjacent layers
        # We iterate through all layers and all comparators in them
        # To avoid infinite loops, we only accept moves that don't increase size
        # (size is fixed in this move, but it might enable future removals)
        # Actually, shifting doesn't change size, so we need a heuristic to accept/reject.
        # We accept if the network remains valid. We might cycle, so we limit attempts.
        
        max_shift_attempts = 50
        shifts_made = 0
        for _ in range(max_shift_attempts):
            if time.time() > deadline - 10:
                break
            
            # Pick a random comparator and try to move it
            if not best_layers:
                break
                
            # Flatten with indices
            flat_with_idx = []
            for l_idx, layer in enumerate(best_layers):
                for c_idx, comp in enumerate(layer):
                    flat_with_idx.append((l_idx, c_idx, comp))
            
            if not flat_with_idx:
                break
                
            l_idx, c_idx, comp = random.choice(flat_with_idx)
            
            # Try moving to previous layer
            if l_idx > 0:
                prev_layer = best_layers[l_idx - 1]
                if can_add_to_layer(comp, prev_layer):
                    # Create candidate
                    new_layers = [list(layer) for layer in best_layers]
                    new_layers[l_idx].pop(c_idx)
                    new_layers[l_idx - 1].append(comp)
                    
                    # Check validity
                    candidate_list = layers_to_list(new_layers)
                    if quick_check(candidate_list):
                        if full_check(candidate_list):
                            best_layers = new_layers
                            shifts_made += 1
                            improved = True
                            break # Restart shifting loop with new structure
            
            # Try moving to next layer
            if l_idx < len(best_layers) - 1:
                next_layer = best_layers[l_idx + 1]
                if can_add_to_layer(comp, next_layer):
                    new_layers = [list(layer) for layer in best_layers]
                    new_layers[l_idx].pop(c_idx)
                    new_layers[l_idx + 1].append(comp)
                    
                    candidate_list = layers_to_list(new_layers)
                    if quick_check(candidate_list):
                        if full_check(candidate_list):
                            best_layers = new_layers
                            shifts_made += 1
                            improved = True
                            break # Restart shifting loop
        
        # Phase 3: Insert-Remove (2-for-1)
        # Try inserting a random comparator, then run a quick removal pass
        if time.time() < deadline - 5:
            current_list = layers_to_list(best_layers)
            # Try a few random insertions
            for _ in range(20):
                if time.time() > deadline - 5:
                    break
                pos = random.randint(0, len(current_list))
                i = random.randint(0, N - 2)
                j = random.randint(i + 1, N - 1)
                candidate = current_list[:pos] + [(i, j)] + current_list[pos:]
                
                # Try to remove 2 comparators from this candidate
                # Simple heuristic: try removing each comparator in the candidate
                # and see if it's valid. If we can remove 2, we win.
                # This is expensive, so we limit it.
                
                # Quick check if candidate is even close to being sortable
                if not quick_check(candidate):
                    continue
                
                # Try to remove one
                removed_one = False
                for k in range(len(candidate)):
                    test = candidate[:k] + candidate[k+1:]
                    if quick_check(test):
                        if full_check(test):
                            # Now try to remove another from 'test'
                            for m in range(len(test)):
                                test2 = test[:m] + test[m+1:]
                                if quick_check(test2):
                                    if full_check(test2):
                                        # Success! We inserted 1 and removed 2.
                                        current_list = test2
                                        best_layers = list_to_layers(current_list)
                                        improved = True
                                        removed_one = True
                                        break
                            if removed_one:
                                break
                    else:
                        continue
                if removed_one:
                    break

    return layers_to_list(best_layers)

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Use odd-even merge sort as the base network
    base = odd_even_merge_sort_network(N)
    
    # Verify base is correct
    if not is_sorted_network(base):
        # Fallback to odd-even transposition
        base = [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    best = list(base)
    best_len = len(best)
    
    # Phase 1: Sequential removal (fast, do it first)
    if time.time() < deadline - 30:
        best = improve_by_removal_sequential(best, deadline)
        best_len = len(best)
    
    # Phase 2: Layer-based local search (structural optimization)
    if time.time() < deadline - 5:
        best = local_search_layers(best, deadline)
        best_len = len(best)
    
    # Phase 3: Final sequential removal to catch any new redundancies from layer shifts
    if time.time() < deadline - 2:
        best = improve_by_removal_sequential(best, deadline)
        best_len = len(best)
    
    # Verify final result
    if not is_sorted_network(best):
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    return best
# EVOLVE-BLOCK-END
