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
_LARGE_SAMPLE_INPUTS = None

def _get_all_inputs():
    global _ALL_INPUTS
    if _ALL_INPUTS is None:
        _ALL_INPUTS = _make_all_inputs()
    return _ALL_INPUTS

def _get_sample_inputs(n_samples=4096):
    global _SAMPLE_INPUTS
    if _SAMPLE_INPUTS is None:
        all_inputs = _get_all_inputs()
        rng = np.random.default_rng(12345)
        idx = rng.choice(len(all_inputs), size=n_samples, replace=False)
        _SAMPLE_INPUTS = all_inputs[idx]
    return _SAMPLE_INPUTS

def _get_large_sample_inputs(n_samples=16384):
    global _LARGE_SAMPLE_INPUTS
    if _LARGE_SAMPLE_INPUTS is None:
        all_inputs = _get_all_inputs()
        rng = np.random.default_rng(99999)
        idx = rng.choice(len(all_inputs), size=n_samples, replace=False)
        _LARGE_SAMPLE_INPUTS = all_inputs[idx]
    return _LARGE_SAMPLE_INPUTS

def simulate_network(comparators, inputs):
    """Simulate compare-exchange network. Returns output array."""
    state = inputs.copy()
    for (i, j) in comparators:
        mi = state[:, i] & state[:, j]
        mj = state[:, i] | state[:, j]
        state[:, i] = mi
        state[:, j] = mj
    return state

def is_sorted_fast(comparators, inputs=None):
    """Quick check on sample inputs."""
    if inputs is None:
        inputs = _get_sample_inputs()
    output = simulate_network(comparators, inputs)
    for k in range(N - 1):
        if np.any(output[:, k] > output[:, k + 1]):
            return False
    return True

def is_sorted_medium(comparators):
    """Medium check on larger sample."""
    inputs = _get_large_sample_inputs()
    output = simulate_network(comparators, inputs)
    for k in range(N - 1):
        if np.any(output[:, k] > output[:, k + 1]):
            return False
    return True

def is_sorted_network(comparators):
    """Full check on all 2^N Boolean inputs."""
    inputs = _get_all_inputs()
    output = simulate_network(comparators, inputs)
    for k in range(N - 1):
        if np.any(output[:, k] > output[:, k + 1]):
            return False
    return True

def odd_even_merge_sort_network(n):
    """
    Generate an odd-even merge sort network for n wires.
    Odd-even merge sort is generally more efficient than odd-even transposition.
    """
    if n <= 1:
        return []
    
    def _odd_even_merge(n, offset=0):
        """Merge two sorted halves of length n/2 starting at offset."""
        if n == 1:
            return []
        comparators = []
        half = n // 2
        # Compare elements at positions offset+i and offset+i+half for i in 0..half-1
        # But we need to handle the merge properly
        # First, compare (offset, offset+half), (offset+1, offset+half+1), ...
        for i in range(half):
            i1 = offset + i
            i2 = offset + i + half
            if i1 < i2:
                comparators.append((i1, i2))
            else:
                comparators.append((i2, i1))
        
        # Now recursively merge the odd and even sub-sequences
        # Even positions: offset, offset+2, offset+4, ...
        # Odd positions: offset+1, offset+3, offset+5, ...
        # We need to merge the two halves of each parity
        
        # For even indices in [offset, offset+n-1]:
        # They are at offset+0, offset+2, offset+4, ...
        # The first half of evens: offset+0, offset+2, ..., offset+half-1 (if half is even) or offset+half-2
        # This gets complex; let's use a different approach
        
        return comparators
    
    def _build(n, offset=0):
        """Build odd-even merge sort network for n wires starting at offset."""
        if n == 1:
            return []
        if n == 2:
            return [(offset, offset + 1)]
        
        comparators = []
        half = n // 2
        # Recursively sort the two halves
        comparators.extend(_build(half, offset))
        comparators.extend(_build(n - half, offset + half))
        
        # Now merge the two sorted halves
        # Use odd-even merge
        comparators.extend(_odd_even_merge_full(n, offset))
        return comparators
    
    def _odd_even_merge_full(n, offset):
        """Full odd-even merge of two sorted sequences of length n/2."""
        if n == 1:
            return []
        if n == 2:
            return [(offset, offset + 1)]
        
        comparators = []
        half = n // 2
        
        # Step 1: Compare (offset+i, offset+i+half) for i = 0, 1, ..., half-1
        for i in range(half):
            i1 = offset + i
            i2 = offset + i + half
            if i1 < i2:
                comparators.append((i1, i2))
            else:
                comparators.append((i2, i1))
        
        # Step 2: Recursively merge the even-indexed and odd-indexed subsequences
        # Even indices: offset, offset+2, offset+4, ...
        # Odd indices: offset+1, offset+3, offset+5, ...
        
        # The even subsequence has elements at positions:
        # offset+0, offset+2, ..., offset+n-2 (if n is even)
        # After the first round of comparisons, the min values go to the first half
        # and max values go to the second half.
        
        # For simplicity, let's use a simpler merge approach
        # Actually, let me implement a proper bitonic-like merge
        
        # Alternative: just use the standard odd-even merge algorithm
        # which involves comparing pairs and then recursively merging
        
        # Let me try a different approach: use the fact that after comparing
        # (i, i+half), the minimums are in the first half and maximums in the second half
        # Then we need to merge the "min" sequence and "max" sequence
        
        # The min sequence consists of elements at positions:
        # min(offset+0, offset+half), min(offset+1, offset+half+1), ...
        # But these aren't necessarily sorted relative to each other
        
        # Let's use a simpler but correct approach: bitonic merge
        return _bitonic_merge_step(n, offset)
    
    def _bitonic_merge_step(n, offset):
        """Perform one level of bitonic merge."""
        if n == 1:
            return []
        comparators = []
        half = n // 2
        for i in range(half):
            i1 = offset + i
            i2 = offset + i + half
            if i1 < i2:
                comparators.append((i1, i2))
            else:
                comparators.append((i2, i1))
        
        # Recursively merge the two halves
        comparators.extend(_bitonic_merge_step(half, offset))
        comparators.extend(_bitonic_merge_step(n - half, offset + half))
        return comparators
    
    return _build(n)

def bitonic_sort_network(n):
    """
    Generate a bitonic sort network for n wires.
    """
    if n == 1:
        return []
    
    def _bitonic_merge(n, offset=0, direction=1):
        # direction 1: ascending, -1: descending
        if n == 1:
            return []
        comparators = []
        # Compare pairs (offset, offset+n/2), (offset+1, offset+n/2+1)...
        half = n // 2
        for i in range(half):
            i1 = offset + i
            i2 = offset + i + half
            if direction == 1:
                if i1 > i2: i1, i2 = i2, i1
            else:
                if i1 < i2: i1, i2 = i2, i1
            comparators.append((i1, i2))
        
        # Recursively merge the two halves
        # Left half: offset .. offset+half-1
        # Right half: offset+half .. offset+n-1
        comparators.extend(_bitonic_merge(half, offset, direction))
        comparators.extend(_bitonic_merge(half, offset + half, direction))
        return comparators

    def _bitonic_sort(n, offset=0, direction=1):
        if n == 1:
            return []
        comparators = []
        half = n // 2
        # Sort left half ascending, right half descending (to form bitonic sequence)
        comparators.extend(_bitonic_sort(half, offset, 1))
        comparators.extend(_bitonic_sort(n - half, offset + half, -1))
        # Merge the bitonic sequence
        comparators.extend(_bitonic_merge(n, offset, direction))
        return comparators

    return _bitonic_sort(n)

def compute_redundancy_scores(comparators):
    """
    For each comparator, compute how many of the sample inputs would fail
    if we removed it. Lower score = more redundant.
    """
    inputs = _get_sample_inputs(8192)
    n_comparators = len(comparators)
    scores = np.zeros(n_comparators, dtype=np.int32)
    
    for idx in range(n_comparators):
        candidate = comparators[:idx] + comparators[idx+1:]
        output = simulate_network(candidate, inputs)
        violations = 0
        for k in range(N - 1):
            violations += np.sum(output[:, k] > output[:, k + 1])
        scores[idx] = violations
    
    return scores

def improve_by_removal_smart(comparators, deadline):
    """
    Use redundancy scores to prioritize which comparators to try removing first.
    """
    best = list(comparators)
    
    while time.time() < deadline - 3:
        if len(best) == 0:
            break
        
        # Compute redundancy scores
        scores = compute_redundancy_scores(best)
        
        # Find the comparator with the lowest violation count
        min_score = np.min(scores)
        
        # If no comparator can be removed without violations on samples, stop
        if min_score > 0:
            # Try all comparators with zero sample violations
            candidates_to_try = np.where(scores == 0)[0]
            if len(candidates_to_try) == 0:
                break
        else:
            candidates_to_try = np.where(scores == min_score)[0]
        
        found_improvement = False
        for idx in candidates_to_try:
            if time.time() > deadline - 3:
                break
            
            candidate = best[:idx] + best[idx+1:]
            
            # Full check
            if is_sorted_network(candidate):
                best = candidate
                found_improvement = True
                break  # Restart scoring after one removal
        
        if not found_improvement:
            # No improvement found with current scoring
            # Try a more aggressive approach: random removal attempts
            attempts = 0
            max_attempts = 20
            while attempts < max_attempts and time.time() < deadline - 3:
                attempts += 1
                idx = random.randint(0, len(best) - 1)
                candidate = best[:idx] + best[idx+1:]
                if is_sorted_fast(candidate):
                    if is_sorted_network(candidate):
                        best = candidate
                        found_improvement = True
                        break
            if not found_improvement:
                break
    
    return best

def try_insert_and_remove(best, deadline):
    """
    Try inserting a random comparator at a random position, then removing redundancies.
    """
    if time.time() > deadline - 5:
        return best
    
    best_len = len(best)
    attempts = 0
    max_attempts = 300
    
    while time.time() < deadline - 5 and attempts < max_attempts:
        attempts += 1
        
        # Insert a random comparator at a random position
        pos = random.randint(0, len(best))
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        candidate = best[:pos] + [(i, j)] + best[pos:]
        
        # Try to remove comparators from the candidate
        k = 0
        improved = True
        while k < len(candidate) and improved:
            improved = False
            if time.time() > deadline - 3:
                break
            test = candidate[:k] + candidate[k+1:]
            if is_sorted_fast(test):
                if is_sorted_network(test):
                    candidate = test
                    improved = True
                    # Don't increment k
                else:
                    k += 1
            else:
                k += 1
        
        if len(candidate) < best_len:
            best = candidate
            best_len = len(best)
    
    return best

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    # Use multiple starting points and pick the best
    base_options = []
    
    # Option 1: Odd-even merge sort
    oems = odd_even_merge_sort_network(N)
    if is_sorted_network(oems):
        base_options.append(oems)
    
    # Option 2: Bitonic sort
    bitonic = bitonic_sort_network(N)
    if is_sorted_network(bitonic):
        base_options.append(bitonic)
    
    # Option 3: Odd-even transposition (fallback)
    oetn = [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]
    base_options.append(oetn)
    
    # Pick the smallest valid base
    base_options.sort(key=len)
    base = base_options[0]
    
    best = list(base)
    best_len = len(best)
    
    # Phase 1: Smart removal using redundancy scores
    if time.time() < deadline - 15:
        best = improve_by_removal_smart(best, deadline)
        best_len = len(best)
    
    # Phase 2: Try insert-then-remove to escape local minima
    if time.time() < deadline - 10:
        best = try_insert_and_remove(best, deadline)
        best_len = len(best)
    
    # Phase 3: Another pass of smart removal
    if time.time() < deadline - 8:
        best = improve_by_removal_smart(best, deadline)
        best_len = len(best)
    
    # Phase 4: Final removal pass with aggressive random attempts
    if time.time() < deadline - 3:
        best = improve_by_removal_smart(best, deadline)
        best_len = len(best)
    
    # Verify final result
    if not is_sorted_network(best):
        # Should not happen, but fallback
        return oetn
    
    return best
# EVOLVE-BLOCK-END
