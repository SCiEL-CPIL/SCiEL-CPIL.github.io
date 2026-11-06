import numpy as np
import time

N = 16

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
    """Precompute all 2^n Boolean inputs as a uint8 matrix (65536 x n)."""
    return (np.arange(1 << n, dtype=np.int32)[:, None] >> np.arange(n, dtype=np.int32)[None, :]) & 1

def _verify(comparators, inputs):
    """Verify network sorts all 2^n Boolean inputs using zero-one principle with NumPy matrix ops."""
    state = inputs.copy()
    for i, j in comparators:
        min_val = np.minimum(state[:, i], state[:, j])
        max_val = np.maximum(state[:, i], state[:, j])
        state[:, i] = min_val
        state[:, j] = max_val
    diffs = np.diff(state, axis=1)
    return bool(np.all(diffs >= 0))

def _remove_redundant(comparators, inputs, time_limit=55):
    """
    Aggressively remove redundant comparators using fast matrix-based verification.
    Multiple phases: single removal (middle-out), pair removal, final single pass.
    """
    start_time = time.time()
    current = list(comparators)
    
    def elapsed():
        return time.time() - start_time
    
    # Phase 1: Greedy single-removal, scanning from middle outward (more likely redundant)
    while True:
        if elapsed() > time_limit - 2:
            break
        
        removed = False
        n_comps = len(current)
        
        if n_comps == 0:
            break
        
        # Build order: middle indices first, then expand outward
        mid = n_comps // 2
        order = list(range(mid, n_comps)) + list(range(mid - 1, -1, -1))
        
        for idx in order:
            if elapsed() > time_limit - 2:
                break
            
            candidate = current[:idx] + current[idx+1:]
            if _verify(candidate, inputs):
                current = candidate
                removed = True
                break
        
        if not removed:
            break
    
    # Phase 2: Try removing pairs of adjacent comparators
    while True:
        if elapsed() > time_limit - 2:
            break
        
        removed = False
        n_comps = len(current)
        
        if n_comps < 2:
            break
        
        # Try removing pairs (i, i+1)
        for i in range(n_comps - 1):
            if elapsed() > time_limit - 2:
                break
            
            candidate = current[:i] + current[i+2:]
            if _verify(candidate, inputs):
                current = candidate
                removed = True
                break
        
        if not removed:
            break
    
    # Phase 3: Final single-removal pass (in case phase 2 opened up new singles)
    while True:
        if elapsed() > time_limit - 2:
            break
        
        removed = False
        n_comps = len(current)
        
        if n_comps == 0:
            break
        
        for idx in range(n_comps):
            if elapsed() > time_limit - 2:
                break
            
            candidate = current[:idx] + current[idx+1:]
            if _verify(candidate, inputs):
                current = candidate
                removed = True
                break
        
        if not removed:
            break
    
    return current

def construct():
    deadline = time.time() + 55
    
    # Generate Batcher's network
    network = _batcher(N)
    
    # Precompute all inputs once for fast verification
    inputs = _precompute_inputs(N)
    
    # Verify initial network
    assert _verify(network, inputs), "Initial Batcher network failed verification"
    
    # Aggressively remove redundant comparators
    network = _remove_redundant(network, inputs, time_limit=55)
    
    # Final safety check
    if time.time() < deadline - 1:
        assert _verify(network, inputs), "Final network failed verification"
    
    return network
