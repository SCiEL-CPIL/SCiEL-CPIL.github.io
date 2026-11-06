"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from collections import defaultdict

N = 17
SEED_COMPARATORS = 136

# Precompute all Boolean inputs and their sorted versions at module load
_NUM_INPUTS = 1 << N
def _make_all_data():
    indices = np.arange(_NUM_INPUTS, dtype=np.uint32)[:, None]
    bit_positions = np.arange(N, dtype=np.uint32)[None, :]
    inputs = ((indices >> bit_positions) & 1).astype(np.uint8)
    sorted_inputs = np.sort(inputs, axis=1)
    return inputs, sorted_inputs

_ALL_INPUTS, _ALL_SORTED = _make_all_data()
_SAMPLE_INPUTS = None
_SAMPLE_SORTED = None

def _get_all_inputs():
    return _ALL_INPUTS

def _get_all_sorted():
    return _ALL_SORTED

def _get_sample_inputs(n_samples=2048):
    global _SAMPLE_INPUTS, _SAMPLE_SORTED
    if _SAMPLE_INPUTS is None:
        rng = np.random.default_rng(12345)
        idx = rng.choice(_NUM_INPUTS, size=n_samples, replace=False)
        _SAMPLE_INPUTS = _ALL_INPUTS[idx]
        _SAMPLE_SORTED = _ALL_SORTED[idx]
    return _SAMPLE_INPUTS, _SAMPLE_SORTED

def simulate_network(comparators, inputs):
    """Simulate compare-exchange network. Returns output array."""
    state = inputs.copy()
    for (i, j) in comparators:
        mi = state[:, i] & state[:, j]
        mj = state[:, i] | state[:, j]
        state[:, i] = mi
        state[:, j] = mj
    return state

def is_sorted_network(comparators):
    """Full check on all 2^N Boolean inputs using precomputed sorted outputs."""
    if len(comparators) > 4096:
        return False
    state = simulate_network(comparators, _ALL_INPUTS)
    return bool(np.array_equal(state, _ALL_SORTED))

def is_sorted_fast(comparators, n_samples=2048):
    """Quick check on sample inputs."""
    sample_in, sample_sorted = _get_sample_inputs(n_samples)
    state = simulate_network(comparators, sample_in)
    return bool(np.array_equal(state, sample_sorted))

def odd_even_merge_sort_network(n):
    """
    Generate a proper odd-even merge sort network for n wires.
    """
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    def _oe_merge(a, b):
        """Odd-even merge of two sorted sequences of lengths a and b."""
        if a == 0:
            return []
        if b == 0:
            return []
        if a == 1 and b == 1:
            return [(0, 1)]
        
        comparators = []
        # First stage: compare (0,1), (2,3), ...
        for i in range(0, a + b - 1, 2):
            comparators.append((i, i + 1))
        
        # Recursive merges on even and odd subsequences
        even_len = (a + b) // 2
        odd_len = (a + b) - even_len
        
        # Even subsequence: positions 0, 2, 4, ...
        even_merge = _oe_merge(a // 2, b // 2)
        for (i, j) in even_merge:
            comparators.append((2 * i, 2 * j))
        
        # Odd subsequence: positions 1, 3, 5, ...
        odd_merge = _oe_merge(a - a // 2, b - b // 2)
        for (i, j) in odd_merge:
            comparators.append((2 * i + 1, 2 * j + 1))
        
        return comparators
    
    def _build(n):
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        left = _build(half)
        right = [(i + half, j + half) for (i, j) in _build(n - half)]
        merged = _oe_merge(half, n - half)
        return left + right + merged
    
    return _build(n)

def compute_redundancy_scores(comparators):
    """
    For each comparator, estimate how 'redundant' it is by checking
    how often it actually changes the output on random inputs.
    Lower score = more redundant = better candidate for removal.
    """
    sample_in, _ = _get_sample_inputs(4096)
    n_comps = len(comparators)
    scores = np.zeros(n_comps, dtype=np.float64)
    
    state = sample_in.copy()
    for ci, (i, j) in enumerate(comparators):
        a = state[:, i]
        b = state[:, j]
        swap_mask = a > b
        scores[ci] = np.sum(swap_mask)
        state[:, i] = np.where(swap_mask, b, a)
        state[:, j] = np.where(swap_mask, a, b)
    
    return scores

def improve_by_removal_sequential(comparators, deadline):
    """
    Sequential pass: try removing each comparator in order.
    Multiple passes. Uses a quick pre-filter with fewer samples.
    """
    best = list(comparators)
    
    sample_in, sample_sorted = _get_sample_inputs(2048)
    
    def quick_check(candidate):
        state = sample_in.copy()
        for (i, j) in candidate:
            mi = state[:, i] & state[:, j]
            mj = state[:, i] | state[:, j]
            state[:, i] = mi
            state[:, j] = mj
        return bool(np.array_equal(state, sample_sorted))
    
    while time.time() < deadline - 3:
        removed_any = False
        i = 0
        while i < len(best):
            if time.time() > deadline - 3:
                break
            candidate = best[:i] + best[i+1:]
            if quick_check(candidate):
                if is_sorted_network(candidate):
                    best = candidate
                    removed_any = True
                    # Don't increment i, try removing next one
                else:
                    i += 1
            else:
                i += 1
        
        if not removed_any:
            break
    
    return best

def improve_by_removal_smart(comparators, deadline):
    """
    Remove comparators in order of redundancy score (most redundant first).
    Multiple passes with re-scoring.
    """
    best = list(comparators)
    
    while time.time() < deadline - 3:
        if len(best) == 0:
            break
        
        scores = compute_redundancy_scores(best)
        order = sorted(range(len(best)), key=lambda x: scores[x])
        
        removed_any = False
        for idx in order:
            if time.time() > deadline - 3:
                break
            if idx >= len(best):
                break
            
            candidate = best[:idx] + best[idx+1:]
            if is_sorted_fast(candidate):
                if is_sorted_network(candidate):
                    best = candidate
                    removed_any = True
                    break
        
        if not removed_any:
            break
    
    return best

def try_insert_and_remove(best, deadline):
    """
    Try inserting a random comparator at a random position, then removing redundancies.
    Helps escape local minima.
    """
    if time.time() > deadline - 3:
        return best
    
    best_len = len(best)
    attempts = 0
    max_attempts = 100
    
    sample_in, sample_sorted = _get_sample_inputs(1024)
    
    def quick_check(candidate):
        state = sample_in.copy()
        for (i, j) in candidate:
            mi = state[:, i] & state[:, j]
            mj = state[:, i] | state[:, j]
            state[:, i] = mi
            state[:, j] = mj
        return bool(np.array_equal(state, sample_sorted))
    
    while time.time() < deadline - 3 and attempts < max_attempts:
        attempts += 1
        
        pos = random.randint(0, len(best))
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        candidate = best[:pos] + [(i, j)] + best[pos:]
        
        k = 0
        improved = True
        while k < len(candidate) and improved:
            improved = False
            if time.time() > deadline - 2:
                break
            test = candidate[:k] + candidate[k+1:]
            if quick_check(test):
                if is_sorted_network(test):
                    candidate = test
                    improved = True
                else:
                    k += 1
            else:
                k += 1
        
        if len(candidate) < best_len:
            best = candidate
            best_len = len(best)
    
    return best

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
            layers.append(current_layer)
            current_layer = []
            used_wires = set()
        current_layer.append((i, j))
        used_wires.add(i)
        used_wires.add(j)
    
    if current_layer:
        layers.append(current_layer)
    
    return layers

def optimize_layer_structure(comparators, deadline):
    """
    Use layer structure to find redundancies.
    """
    best = list(comparators)
    
    sample_in, sample_sorted = _get_sample_inputs(2048)
    
    def quick_check(candidate):
        state = sample_in.copy()
        for (i, j) in candidate:
            mi = state[:, i] & state[:, j]
            mj = state[:, i] | state[:, j]
            state[:, i] = mi
            state[:, j] = mj
        return bool(np.array_equal(state, sample_sorted))
    
    while time.time() < deadline - 3:
        layers = build_layers(best)
        removed_any = False
        
        for layer_idx in range(len(layers)):
            if time.time() > deadline - 3:
                break
            for comp_idx in range(len(layers[layer_idx])):
                if time.time() > deadline - 3:
                    break
                global_idx = sum(len(layers[l]) for l in range(layer_idx)) + comp_idx
                if global_idx >= len(best):
                    continue
                
                candidate = best[:global_idx] + best[global_idx+1:]
                if quick_check(candidate):
                    if is_sorted_network(candidate):
                        best = candidate
                        removed_any = True
                        break
            if removed_any:
                break
        
        if not removed_any:
            break
    
    return best

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Use odd-even merge sort as the base network
    base = odd_even_merge_sort_network(N)
    
    # Verify base is correct; if not, fall back to odd-even transposition
    if not is_sorted_network(base):
        base = [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    best = list(base)
    best_len = len(best)
    
    # Phase 1: Sequential removal (most effective, do it first)
    if time.time() < deadline - 25:
        best = improve_by_removal_sequential(best, deadline)
        best_len = len(best)
    
    # Phase 2: Smart removal based on redundancy scores
    if time.time() < deadline - 18:
        best = improve_by_removal_smart(best, deadline)
        best_len = len(best)
    
    # Phase 3: Layer-structure optimization
    if time.time() < deadline - 12:
        best = optimize_layer_structure(best, deadline)
        best_len = len(best)
    
    # Phase 4: Another sequential pass to catch remaining redundancies
    if time.time() < deadline - 8:
        best = improve_by_removal_sequential(best, deadline)
        best_len = len(best)
    
    # Phase 5: Try insert-then-remove to escape local minima
    if time.time() < deadline - 5:
        best = try_insert_and_remove(best, deadline)
        best_len = len(best)
    
    # Phase 6: Final sequential removal
    if time.time() < deadline - 2:
        best = improve_by_removal_sequential(best, deadline)
        best_len = len(best)
    
    # Verify final result
    if not is_sorted_network(best):
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    return best
# EVOLVE-BLOCK-END
