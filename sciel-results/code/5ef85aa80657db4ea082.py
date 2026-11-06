import numpy as np
import time

N = 16
TOTAL_INPUTS = 1 << N
MASK_ALL = TOTAL_INPUTS - 1

def _merge(k):
    """
    Generates comparators to merge two sorted lists of length k.
    """
    if k == 1:
        return [(0, 1)]
    
    k_half = k // 2
    
    even_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i if i < k_half else 2 * (i - k_half) + k
        mj = 2 * j if j < k_half else 2 * (j - k_half) + k
        even_comps.append((mi, mj))
        
    odd_comps = []
    for i, j in _merge(k_half):
        mi = 2 * i + 1 if i < k_half else 2 * (i - k_half) + k + 1
        mj = 2 * j + 1 if j < k_half else 2 * (j - k_half) + k + 1
        odd_comps.append((mi, mj))
        
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

def _simulate_full(network, inputs):
    """
    Simulate the network on the given inputs (numpy array of int32 masks).
    Returns the final masks array.
    """
    masks = inputs.copy()
    for i, j in network:
        bit_i = (masks >> i) & 1
        bit_j = (masks >> j) & 1
        swap = bit_i & (~bit_j) & 1
        if np.any(swap):
            masks = np.where(swap, (masks & ~(1 << i)) | (1 << j), masks)
    return masks

def _is_sorted_array(masks):
    """Check which masks in the array are sorted. Returns boolean array."""
    inv = (~masks) & MASK_ALL
    return (inv & (inv + 1)) == 0

def _verify_full(network, inputs):
    """Full verification on all inputs. Returns True if network sorts all."""
    final = _simulate_full(network, inputs)
    return np.all(_is_sorted_array(final))

def _find_redundant_single(network, inputs, deadline):
    """
    Try to find a single redundant comparator.
    Returns the index if found, else None.
    """
    n_comps = len(network)
    for idx in range(n_comps):
        if time.time() > deadline:
            return None
        # Create modified network
        new_network = network[:idx] + network[idx+1:]
        if _verify_full(new_network, inputs):
            return idx
    return None

def _find_redundant_pair(network, inputs, deadline):
    """
    Try to find a pair of redundant comparators.
    Returns (i, j) if found, else None.
    """
    n_comps = len(network)
    if n_comps < 2:
        return None
    
    # Try adjacent pairs first (more likely to be redundant in sorting networks)
    for i in range(n_comps - 1):
        if time.time() > deadline:
            return None
        new_network = network[:i] + network[i+2:]
        if _verify_full(new_network, inputs):
            return (i, i+1)
    
    # Then try all pairs (limit search for time)
    for i in range(n_comps):
        if time.time() > deadline:
            return None
        for j in range(i + 1, n_comps):
            if time.time() > deadline:
                return None
            # Skip if already tried adjacent
            if j == i + 1:
                continue
            new_network = network[:i] + network[i+1:j] + network[j+1:]
            if _verify_full(new_network, inputs):
                return (i, j)
    return None

def _remove_redundant(network, inputs, deadline):
    """
    Iteratively remove redundant comparators.
    """
    # Phase 1: Single removals
    while time.time() < deadline:
        idx = _find_redundant_single(network, inputs, deadline)
        if idx is not None:
            network = network[:idx] + network[idx+1:]
        else:
            break
    
    # Phase 2: Pair removals
    while time.time() < deadline:
        pair = _find_redundant_pair(network, inputs, deadline)
        if pair is not None:
            i, j = pair
            if i > j:
                i, j = j, i
            # Remove j first, then i (since j > i, removing j shifts indices after j)
            # Actually, safer to reconstruct
            new_network = network[:i] + network[i+1:j] + network[j+1:]
            network = new_network
        else:
            break
    
    # Phase 3: More single removals after pair removals
    while time.time() < deadline:
        idx = _find_redundant_single(network, inputs, deadline)
        if idx is not None:
            network = network[:idx] + network[idx+1:]
        else:
            break
    
    return network

def construct():
    deadline = time.time() + 85  # 85s to leave 5s margin
    
    # Start with Batcher's Odd-Even Merge Sort (63 comparators for n=16)
    network = _batcher(N)
    
    # Precompute all inputs
    inputs = np.arange(TOTAL_INPUTS, dtype=np.int32)
    
    # Verify initial network
    assert _verify_full(network, inputs), "Initial Batcher network should sort all inputs"
    
    # Remove redundant comparators
    network = _remove_redundant(network, inputs, deadline)
    
    return network
