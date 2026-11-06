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

def _get_sample_inputs(n_samples=8192):
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

def is_sorted_fast(comparators, inputs=None):
    """Quick check on sample inputs."""
    if inputs is None:
        inputs = _get_sample_inputs()
    output = simulate_network(comparators, inputs)
    # Vectorized sorted check
    return np.all(output[:, :-1] <= output[:, 1:])

def is_sorted_network(comparators):
    """Full check on all 2^N Boolean inputs."""
    inputs = _get_all_inputs()
    output = simulate_network(comparators, inputs)
    # Vectorized sorted check
    return np.all(output[:, :-1] <= output[:, 1:])

def find_redundant_comparators(comparators):
    """
    Find comparators that are no-ops on ALL Boolean inputs.
    A comparator (i,j) is redundant if state[i] <= state[j] for all inputs
    at the point it's reached. Returns list of indices (0-based) of redundant comparators.
    """
    inputs = _get_all_inputs()
    state = inputs.copy()
    redundant = []
    for idx, (i, j) in enumerate(comparators):
        # Check if state[i] <= state[j] for all rows
        # i.e., state[i] & ~state[j] should be all zeros
        if np.any(state[:, i] & ~state[:, j]):
            # Not redundant, apply comparator
            mi = state[:, i] & state[:, j]
            mj = state[:, i] | state[:, j]
            state[:, i] = mi
            state[:, j] = mj
        else:
            # Redundant: state[i] <= state[j] everywhere
            redundant.append(idx)
            # Still "apply" (no-op)
    return redundant

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
        # Recursively sort first half and second half (offset)
        left = _build(half)
        right = [(i + half, j + half) for (i, j) in _build(n - half)]
        
        # Odd-even merge
        merged = _odd_even_merge(n)
        
        return left + right + merged
    
    def _odd_even_merge(n):
        if n <= 1:
            return []
        if n == 2:
            return [(0, 1)]
        
        half = n // 2
        comparators = []
        
        if n % 2 == 0:
            # Even case: compare adjacent pairs starting from 0
            for i in range(0, n, 2):
                comparators.append((i, i + 1))
            # Now recursively merge: even positions form one sub-problem, odd form another
            even_merged = _odd_even_merge(n // 2)
            for (a, b) in even_merged:
                comparators.append((2 * a, 2 * b))
            odd_merged = _odd_even_merge(n // 2)
            for (a, b) in odd_merged:
                comparators.append((2 * a + 1, 2 * b + 1))
        else:
            # Odd case: compare (1,2),(3,4),...
            for i in range(1, n - 1, 2):
                comparators.append((i, i + 1))
            # Element 0 stays, element n-1 stays
            # Recursively merge the middle: positions 1..n-2 (n-2 elements)
            if n - 2 > 0:
                mid_merged = _odd_even_merge(n - 2)
                for (a, b) in mid_merged:
                    comparators.append((a + 1, b + 1))
        
        return comparators
    
    return _build(n)

def odd_even_transposition_network(n):
    """Generate odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(n)
            for i in range(phase % 2, n - 1, 2)]

def improve_by_removal(comparators, deadline):
    """
    Iteratively try to remove each comparator if the network remains correct.
    First pass: bulk-remove provably redundant comparators (no-ops on all inputs).
    Second pass: sequential removal with fast+full validation.
    """
    best = list(comparators)
    
    # Phase A: Bulk remove no-op comparators (fast, single simulation)
    while time.time() < deadline - 5:
        redundant = find_redundant_comparators(best)
        if not redundant:
            break
        # Remove all redundant at once (they're independent no-ops)
        redundant_set = set(redundant)
        best = [best[k] for k in range(len(best)) if k not in redundant_set]
    
    # Phase B: Sequential removal with validation
    # We use a while loop that resets i to 0 whenever a removal succeeds,
    # to catch new redundancies created by previous removals.
    while time.time() < deadline - 3:
        i = 0
        removed_any = False
        while i < len(best):
            if time.time() > deadline - 3:
                break
            candidate = best[:i] + best[i+1:]
            # Fast check first
            if is_sorted_fast(candidate):
                # Full check
                if is_sorted_network(candidate):
                    best = candidate
                    removed_any = True
                    # Do not increment i, re-check same position
                else:
                    i += 1
            else:
                i += 1
        if not removed_any:
            break
            
    return best

def try_insert_and_remove(best, deadline, max_attempts=500):
    """
    Try inserting a random comparator at a random position, then removing redundancies.
    This can escape local minima. Uses bulk no-op removal for speed.
    """
    if time.time() > deadline - 5:
        return best
    
    best_len = len(best)
    attempts = 0
    
    while time.time() < deadline - 5 and attempts < max_attempts:
        attempts += 1
        
        # Insert a random comparator at a random position
        pos = random.randint(0, len(best))
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        candidate = best[:pos] + [(i, j)] + best[pos:]
        
        # Bulk remove no-ops
        redundant = find_redundant_comparators(candidate)
        if redundant:
            redundant_set = set(redundant)
            candidate = [candidate[k] for k in range(len(candidate)) if k not in redundant_set]
        
        # Sequential removal
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
                else:
                    k += 1
            else:
                k += 1
        
        if len(candidate) < best_len:
            best = candidate
            best_len = len(best)
    
    return best

def swap_comparators(comparators, deadline, max_attempts=200):
    """
    Try swapping two comparators in the network to see if it enables further removal.
    Uses bulk no-op removal for speed.
    """
    best = list(comparators)
    best_len = len(best)
    
    for _ in range(max_attempts):
        if time.time() > deadline - 5:
            break
        
        # Pick two random positions to swap
        if len(best) < 2:
            break
        p1 = random.randint(0, len(best) - 1)
        p2 = random.randint(0, len(best) - 1)
        if p1 == p2:
            continue
        
        candidate = best[:]
        candidate[p1], candidate[p2] = candidate[p2], candidate[p1]
        
        # Bulk remove no-ops
        redundant = find_redundant_comparators(candidate)
        if redundant:
            redundant_set = set(redundant)
            candidate = [candidate[k] for k in range(len(candidate)) if k not in redundant_set]
        
        # Sequential removal
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
    
    # Generate base networks
    base_merge = odd_even_merge_sort_network(N)
    base_trans = odd_even_transposition_network(N)
    
    # Verify bases are correct using fast check first to save time if invalid
    base_to_use = base_trans
    if is_sorted_fast(base_merge):
        if is_sorted_network(base_merge):
            base_to_use = base_merge
    
    best = list(base_to_use)
    best_len = len(best)
    
    # Phase 1: Aggressive removal (bulk no-ops + sequential)
    if time.time() < deadline - 15:
        best = improve_by_removal(best, deadline)
        best_len = len(best)
    
    # Phase 2: Insert-then-remove to escape local minima
    if time.time() < deadline - 12:
        best = try_insert_and_remove(best, deadline, max_attempts=600)
        best_len = len(best)
    
    # Phase 3: Swap to enable further removal
    if time.time() < deadline - 8:
        best = swap_comparators(best, deadline, max_attempts=300)
        best_len = len(best)
    
    # Phase 4: Another aggressive removal pass
    if time.time() < deadline - 5:
        best = improve_by_removal(best, deadline)
        best_len = len(best)
    
    # Phase 5: Final removal pass
    if time.time() < deadline - 2:
        best = improve_by_removal(best, deadline)
        best_len = len(best)
    
    # Verify final result
    if not is_sorted_network(best):
        # Fallback to the verified base
        return base_to_use
    
    return best
# EVOLVE-BLOCK-END
