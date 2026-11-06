"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START

def _merge(k):
    """
    Generates comparators to merge two sorted lists of length k.
    The lists are assumed to be at indices 0..k-1 and k..2k-1.
    """
    if k == 1:
        return [(0, 1)]
    
    k_half = k // 2
    
    # 1. Merge even-indexed elements
    even_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i if i < k_half else 2 * (i - k_half) + k
        mj = 2 * j if j < k_half else 2 * (j - k_half) + k
        even_comps.append((mi, mj))
        
    # 2. Merge odd-indexed elements
    odd_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i + 1 if i < k_half else 2 * (i - k_half) + k + 1
        mj = 2 * j + 1 if j < k_half else 2 * (j - k_half) + k + 1
        odd_comps.append((mi, mj))
        
    # 3. Final comparisons between adjacent pairs (2i+1, 2i+2)
    final_comps = [(2 * i + 1, 2 * i + 2) for i in range(k - 1)]
    
    return even_comps + odd_comps + final_comps

def _batcher(n):
    """
    Generates Batcher's Odd-Even Merge Sort network for n wires.
    """
    if n == 1:
        return []
    
    k = n // 2
    left = _batcher(k)
    right = [(i + k, j + k) for i, j in _batcher(k)]
    merge = _merge(k)
    
    return left + right + merge

def _precompute_inputs(n=N):
    """Precompute all 2^n Boolean inputs as a uint8 matrix for fast verification."""
    # Using int32 for safety in shifts, then converting to uint8 for storage if needed, 
    # but int32/uint8 arithmetic is fine. Let's use uint8 for the final state to be memory efficient.
    # Actually, np.arange(1<<n) is int64 by default on 64-bit.
    # We want bits.
    return (np.arange(1 << n, dtype=np.int32)[:, None] >> np.arange(n, dtype=np.int32)[None, :]) & 1

def _verify_network(comparators, n=N, inputs=None):
    """
    Verify network sorts all 2^n Boolean inputs using zero-one principle with NumPy.
    Returns True if valid.
    """
    if inputs is None:
        inputs = _precompute_inputs(n)
    
    # Work on a copy. inputs is int32 usually.
    state = inputs.copy()
    
    for i, j in comparators:
        # min goes to i, max goes to j
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
        
    # Check if sorted: each row should be non-decreasing
    # diffs = state[:, 1:] - state[:, :-1]
    # return bool(np.all(diffs >= 0))
    
    # Faster check: count of 1s must match input count? No, we need order.
    # A sorted row of 0s and 1s looks like 000...011...1
    # So state[:, i] <= state[:, i+1]
    return bool(np.all(state[:, :-1] <= state[:, 1:]))

def _remove_redundant(comparators, n=N, time_limit=85.0):
    """
    Aggressively remove redundant comparators.
    Strategy:
    1. Precompute inputs once.
    2. Greedy single-removal with randomized order to escape local minima.
    3. If stuck, try limited pair-removal (removing two comparators at once).
    """
    start_time = time.time()
    current = list(comparators)
    inputs = _precompute_inputs(n)
    
    def elapsed():
        return time.time() - start_time

    while True:
        if elapsed() > time_limit:
            break
        
        removed = False
        n_comps = len(current)
        
        if n_comps == 0:
            break
        
        # Randomize order to explore different local minima
        order = list(range(n_comps))
        random.shuffle(order)
        
        for idx in order:
            if elapsed() > time_limit:
                break
            
            if idx >= len(current):
                continue
                
            candidate = current[:idx] + current[idx+1:]
            
            if _verify_network(candidate, n, inputs):
                current = candidate
                removed = True
                break
        
        if not removed:
            # Try limited pair removal if single removal fails
            # We look for pairs (i, j) that can be removed together.
            # To keep it fast, we only check pairs within a small window.
            n_comps = len(current)
            if n_comps < 2:
                break
                
            pair_removed = False
            window = 10  # Check pairs within 10 indices of each other
            
            for i in range(n_comps):
                if elapsed() > time_limit:
                    break
                # j must be > i
                max_j = min(i + 1 + window, n_comps)
                for j in range(i + 1, max_j):
                    # Remove i and j
                    # Indices: 0 .. i-1, i, i+1 .. j-1, j, j+1 ..
                    # New list: 0 .. i-1, i+1 .. j-1, j+1 ..
                    candidate = current[:i] + current[i+1:j] + current[j+1:]
                    
                    if _verify_network(candidate, n, inputs):
                        current = candidate
                        pair_removed = True
                        break
                if pair_removed:
                    break
            
            if not pair_removed:
                # If we can't remove singles or nearby pairs, we are likely at a deep local minimum.
                # We could try a larger window or random pairs, but it gets expensive.
                # For now, stop.
                break
                
    return current

def construct():
    """
    Construct the sorting network.
    """
    # Generate Batcher's network
    # Batcher OES for 16 wires has 120 comparators.
    network = _batcher(N)
    
    # Verify base network just in case
    if not _verify_network(network, N):
        # Fallback to odd-even transposition if Batcher generation failed (should not happen)
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Apply local search (redundancy removal) with time budget
    # The evaluator kills at 90s. We use 85s to maximize search time.
    result = _remove_redundant(network, N, time_limit=85.0)
    
    # Final verification
    if not _verify_network(result, N):
        return network
    
    return result

# EVOLVE-BLOCK-END
