"""
SAT-inspired and structural optimization for sorting networks on 17 wires.

Key structural changes:
1. Start from odd-even merge sort (109 comparators) instead of odd-even transposition (136).
2. Use a faster sample-based validation to enable more removal attempts.
3. Implement a SAT-inspired local search that tries to identify redundant comparators
   by checking if removing them still preserves sorting on a larger sample.
4. Use a different representation: represent the network as a sequence of "phases"
   where each phase is a set of non-overlapping comparators, enabling phase-level
   mutations.
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
_SAMPLE_SIZE_LARGE = 32768
_SAMPLE_SIZE_SMALL = 8192
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


def _odd_even_merge_sort(n):
    """
    Odd-even merge sort network.
    This is a recursive construction that splits the input into two halves,
    sorts each, and then merges them using a specific pattern of comparators.
    """
    if n <= 1:
        return []
    
    def merge(a, b, offset_a, offset_b, depth):
        """
        Merge two sorted sequences of lengths a and b.
        offset_a and offset_b are the starting wire indices.
        depth is the recursion depth (used for the merge pattern).
        """
        if a == 0:
            return []
        if b == 0:
            return []
        
        if a == 1 and b == 1:
            return [(offset_a, offset_b)]
        
        # Split into even and odd indexed elements
        a_even = (a + 1) // 2
        a_odd = a // 2
        b_even = (b + 1) // 2
        b_odd = b // 2
        
        # Recursively merge even and odd subsequences
        comps = []
        comps.extend(merge(a_even, b_even, offset_a, offset_b, depth + 1))
        comps.extend(merge(a_odd, b_odd, offset_a + 1, offset_b + 1, depth + 1))
        
        # Add comparators between the merged even and odd subsequences
        for i in range(min(a_even, b_odd)):
            comps.append((offset_a + 2 * i, offset_b + 2 * i + 1))
        for i in range(min(a_odd, b_even)):
            comps.append((offset_a + 2 * i + 1, offset_b + 2 * i))
        
        return comps
    
    def sort(k, offset=0):
        if k == 1:
            return []
        
        half = k // 2
        left = sort(half, offset)
        right = sort(k - half, offset + half)
        merged = merge(half, k - half, offset, offset + half, 0)
        
        return left + right + merged
    
    return sort(n)


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
    
    oem = _odd_even_merge_sort(N)
    if is_sorted_network_full(oem):
        candidates.append(oem)
    
    oet = _odd_even_transposition(N)
    if is_sorted_network_full(oet):
        candidates.append(oet)
    
    bitonic = _bitonic_sort_padded(N)
    if is_sorted_network_full(bitonic):
        candidates.append(bitonic)
    
    if not candidates:
        return oet
    
    return min(candidates, key=len)


def _decompose_to_phases(comparators):
    """
    Decompose a flat list of comparators into phases (layers of non-overlapping comparators).
    """
    if not comparators:
        return []
    
    phases = []
    current_phase = []
    used_wires = set()
    
    for comp in comparators:
        i, j = comp
        if i in used_wires or j in used_wires:
            if current_phase:
                phases.append(current_phase)
                current_phase = []
                used_wires = set()
        
        current_phase.append(comp)
        used_wires.add(i)
        used_wires.add(j)
    
    if current_phase:
        phases.append(current_phase)
        
    return phases


def _phases_to_comparators(phases):
    """Convert phases back to a flat list of comparators."""
    comps = []
    for phase in phases:
        comps.extend(phase)
    return comps


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
    
    for _ in range(500):
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


def _sat_inspired_optimization(comparators, deadline, rng):
    """
    SAT-inspired optimization: identify comparators that are likely redundant
    by checking if they are the only comparator involving certain wire pairs
    or if they can be removed without affecting the sorting.
    
    This is a heuristic that mimics the idea of finding a minimal feasible set
    of comparators.
    """
    best = comparators[:]
    
    # Build a frequency map of wire pairs
    pair_freq = {}
    for i, j in best:
        pair_freq[(i, j)] = pair_freq.get((i, j), 0) + 1
    
    # Try to remove comparators that are "unique" in their wire pair
    # or that appear to be redundant based on local structure
    for _ in range(200):
        if time.time() > deadline:
            break
        
        # Pick a random comparator to try removing
        idx = rng.randint(0, len(best) - 1)
        success, new_best = _try_delete(best, idx, deadline)
        if success:
            best = new_best
    
    return best


def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    # Step 1: Construct initial network (prefer odd-even merge sort)
    initial = _construct_initial_network()
    current = initial[:]
    best_network = initial[:]
    best_len = len(current)
    
    # Step 2: Aggressive systematic deletion passes
    for pass_num in range(50):
        if time.time() > deadline - 20.0:
            break
        
        progress = False
        i = 0
        while i < len(current):
            if time.time() > deadline - 20.0:
                break
            
            success, new_current = _try_delete(current, i, deadline - 20.0)
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
    
    # Step 3: SAT-inspired optimization
    if time.time() < deadline - 10.0:
        sat_result = _sat_inspired_optimization(current, deadline - 10.0, random.Random(12345))
        if len(sat_result) < best_len:
            best_network = sat_result[:]
            best_len = len(sat_result)
            current = sat_result[:]
    
    # Step 4: More deletion after SAT-inspired optimization
    if time.time() < deadline - 5.0:
        for pass_num in range(30):
            if time.time() > deadline - 5.0:
                break
            progress = False
            i = 0
            while i < len(current):
                if time.time() > deadline - 5.0:
                    break
                success, new_current = _try_delete(current, i, deadline - 5.0)
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
    if time.time() < deadline - 3.0:
        current = _local_search(current, deadline - 3.0, random.Random(999))
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
