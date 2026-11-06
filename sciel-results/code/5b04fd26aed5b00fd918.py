import time
import numpy as np

N = 18

# EVOLVE-BLOCK-START

def _odd_even_transposition_network(n):
    """Generate an odd-even transposition sorting network for n wires."""
    comparators = []
    for phase in range(n):
        for i in range(phase % 2, n - 1, 2):
            comparators.append((i, i + 1))
    return comparators


def _verify_network_fast(comparators, n=N):
    """
    Verify that the network sorts all boolean inputs using vectorized NumPy.
    """
    if not comparators:
        return n <= 1
    
    for comp in comparators:
        if comp[0] >= comp[1] or comp[0] < 0 or comp[1] >= n:
            return False

    # Generate all 2^n boolean inputs
    masks = np.arange(1 << n, dtype=np.uint32)
    # Create input matrix of shape (2^n, n)
    # Ensure shift is compatible
    shifts = np.arange(n, dtype=np.uint32)
    M = ((masks[:, None] >> shifts[None, :]) & 1).astype(np.int8)

    # Apply comparators
    for i, j in comparators:
        min_vals = np.minimum(M[:, i], M[:, j])
        max_vals = np.maximum(M[:, i], M[:, j])
        M[:, i] = min_vals
        M[:, j] = max_vals

    # Check if sorted ascending: [0, 0, ..., 1, 1, ...]
    # This means M[k, i] <= M[k, i+1] for all i
    diffs = M[:, :-1] - M[:, 1:]
    if np.any(diffs < 0):
        return False
        
    return True


def _batcher_merge(a, b, offset):
    """
    Generate comparators to merge two sorted sequences of lengths a and b
    starting at 'offset'. Uses OET on the combined block for correctness.
    """
    if a == 0 or b == 0:
        return []
    
    combined_len = a + b
    comps = []
    for phase in range(combined_len):
        for i in range(phase % 2, combined_len - 1, 2):
            p1 = offset + i
            p2 = offset + i + 1
            if p1 < offset + combined_len and p2 < offset + combined_len:
                if p1 != p2:
                    comps.append((p1, p2))
    return comps


def _batcher_sort(n, offset=0):
    """
    Generate a Batcher sorting network for n wires starting at 'offset'.
    """
    if n <= 1:
        return []
    if n == 2:
        return [(offset, offset + 1)]
    
    m = n // 2
    rest = n - m
    
    comparators = []
    comparators.extend(_batcher_sort(m, offset))
    comparators.extend(_batcher_sort(rest, offset + m))
    comparators.extend(_batcher_merge(m, rest, offset))
    
    return comparators


def construct():
    """
    Construct a sorting network for N=18 wires.
    Returns an ordered list of integer pairs (i, j) where i < j.
    """
    start_time = time.time()
    deadline = start_time + 85  # Leave 5s margin
    
    # Start with Batcher's network (using OET for merge steps)
    network = _batcher_sort(N, 0)
    
    # Verify it's correct
    if not _verify_network_fast(network, N):
        # Fallback to odd-even transposition
        network = _odd_even_transposition_network(N)
        if not _verify_network_fast(network, N):
            return network

    # Optimize by removing redundant comparators
    optimized = list(network)
    
    # Greedy deletion loop
    # We will try to remove comparators in reverse order, then random order.
    # Multiple passes to ensure local minima are escaped somewhat.
    
    while time.time() < deadline - 5:
        changed = False
        
        # Pass 1: Reverse order
        for idx in range(len(optimized) - 1, -1, -1):
            if time.time() > deadline - 5:
                break
            if idx >= len(optimized):
                continue
            candidate = optimized[:idx] + optimized[idx+1:]
            if _verify_network_fast(candidate, N):
                optimized = candidate
                changed = True
        
        # Pass 2: Random order
        if time.time() < deadline - 5:
            indices = list(range(len(optimized)))
            np.random.shuffle(indices)
            for idx in indices:
                if time.time() > deadline - 5:
                    break
                if idx >= len(optimized):
                    continue
                candidate = optimized[:idx] + optimized[idx+1:]
                if _verify_network_fast(candidate, N):
                    optimized = candidate
                    changed = True
        
        if not changed:
            break

    # Final verification
    if not _verify_network_fast(optimized, N):
        return network
    
    return optimized

# EVOLVE-BLOCK-END
