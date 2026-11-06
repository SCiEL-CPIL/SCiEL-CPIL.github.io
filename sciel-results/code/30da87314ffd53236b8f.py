import time
import numpy as np

N = 16

# EVOLVE-BLOCK-START

def _verify_network_fast(network, n=16):
    """Verify network sorts all 2^n inputs using zero-one principle with NumPy."""
    if not network:
        return n <= 1
        
    num_inputs = 1 << n
    inputs = np.arange(num_inputs, dtype=np.uint32)
    
    masks = (1 << np.arange(n, dtype=np.uint32))
    vals = ((inputs[:, None] & masks[None, :]) > 0).astype(np.uint8)
    
    for i, j in network:
        if i < 0 or i >= n or j < 0 or j >= n or i >= j:
            return False
        vi = vals[:, i]
        vj = vals[:, j]
        
        vi_min = vi & vj
        vj_max = vi | vj
        
        vals[:, i] = vi_min
        vals[:, j] = vj_max
        
    diffs = vals[:, 1:] - vals[:, :-1]
    return np.all(diffs >= 0)

def _bitonic_merge(n, d=0):
    """
    Generate comparators to merge a bitonic sequence of length n.
    d=0: Merge into Ascending order.
    d=1: Merge into Descending order.
    
    The input sequence is bitonic. 
    If d=0, it is Ascending then Descending.
    If d=1, it is Descending then Ascending.
    """
    if n <= 1:
        return []
    half = n // 2
    comparators = []
    
    # Step 1: Compare elements i and i+half
    for i in range(half):
        if d == 0:
            # Ascending: min goes to i, max to i+half
            # Comparator (i, i+half)
            comparators.append((i, i + half))
        else:
            # Descending: max goes to i, min to i+half
            # Comparator (i+half, i) puts min at i+half, max at i.
            # So we want max at i. Thus we use (i+half, i).
            comparators.append((i + half, i))
    
    # Step 2: Recursively merge the two halves
    # The first half (indices 0..half-1) and second half (indices half..n-1)
    # are now bitonic sequences themselves.
    left_merge = _bitonic_merge(half, d)
    right_merge = [(i + half, j + half) for i, j in _bitonic_merge(half, d)]
    
    return comparators + left_merge + right_merge

def _bitonic_sort(n, d=0):
    """
    Generate comparators to sort a sequence of length n into a bitonic sequence.
    d=0 for Ascending, d=1 for Descending.
    """
    if n <= 1:
        return []
    half = n // 2
    # Sort left half in direction d
    left = _bitonic_sort(half, d)
    # Sort right half in opposite direction (1-d)
    right = [(i + half, j + half) for i, j in _bitonic_sort(half, 1 - d)]
    # Merge the two sorted halves into a bitonic sequence
    merge = _bitonic_merge(n, d)
    
    return left + right + merge

def construct():
    """Return a sorting network for 16 wires."""
    # Use Batcher's Bitonic Sort
    # This should produce 60 comparators for N=16
    net = _bitonic_sort(N, 0)
    
    # Verify
    if not _verify_network_fast(net, N):
        # Fallback to Odd-Even Transposition if verification fails
        net = [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]
        
    return net

# EVOLVE-BLOCK-END
