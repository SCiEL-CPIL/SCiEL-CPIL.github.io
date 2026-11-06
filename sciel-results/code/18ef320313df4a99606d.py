"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools

N = 16

# EVOLVE-BLOCK-START
import numpy as np

def _verify_network_fast(network, n=16):
    """Verify network sorts all 2^n inputs using zero-one principle, vectorized."""
    # All 2^n bit patterns as a 2D array of shape (2^n, n)
    total = 1 << n
    # Generate all bit patterns
    bits = np.arange(total, dtype=np.uint8).reshape(-1, 1)
    # Each column is a bit position
    patterns = np.zeros((total, n), dtype=np.uint8)
    for k in range(n):
        patterns[:, k] = ((bits >> k) & 1).astype(np.uint8)
    
    for i, j in network:
        # For each input, compare wire i and wire j
        # If patterns[:, i] > patterns[:, j], swap them
        # min goes to i, max goes to j
        a = patterns[:, i]
        b = patterns[:, j]
        # min = (a + b) / 2 - |a - b| / 2, but for 0/1: min = a*b (AND), max = a|b (OR)
        # Actually for 0/1: min(a,b) = a & b, max(a,b) = a | b
        patterns[:, i] = a & b
        patterns[:, j] = a | b
    
    # Check if sorted: each row should be non-decreasing
    # sorted means: for all k, patterns[:, k] <= patterns[:, k+1]
    # i.e., patterns[:, k] - patterns[:, k+1] <= 0
    # For 0/1: patterns[:, k] > patterns[:, k+1] iff patterns[:, k] == 1 and patterns[:, k+1] == 0
    diffs = patterns[:, :-1] - patterns[:, 1:]
    # Any row with a negative diff is unsorted
    if np.any(diffs < 0):
        return False
    return True

def _batcher_odd_even_mergesort(n):
    """Generate Batcher's odd-even mergesort network for n wires."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    if n == 4:
        return [(0, 1), (2, 3), (0, 2), (1, 3), (1, 2)]
    
    half = n // 2
    left = _batcher_odd_even_mergesort(half)
    right = [(i + half, j + half) for i, j in _batcher_odd_even_mergesort(half)]
    
    merge = [(i, i + half) for i in range(half)]
    
    even_subseq = _batcher_odd_even_mergesort(half)
    even_mapped = [(2 * a, 2 * b) for a, b in even_subseq]
    
    odd_subseq = _batcher_odd_even_mergesort(half)
    odd_mapped = [(2 * a + 1, 2 * b + 1) for a, b in odd_subseq]
    
    return left + right + merge + even_mapped + odd_mapped

def _try_prune(network, n=16, deadline=None):
    """Try to remove redundant comparators from a known-correct network."""
    if deadline is not None and time.time() > deadline - 5:
        return network
    
    current = list(network)
    improved = True
    while improved:
        if deadline is not None and time.time() > deadline - 5:
            break
        improved = False
        for idx in range(len(current)):
            if deadline is not None and time.time() > deadline - 5:
                break
            candidate = current[:idx] + current[idx+1:]
            if _verify_network_fast(candidate, n):
                current = candidate
                improved = True
                break
    return current

def construct():
    """Return a sorting network for 16 wires, minimized in comparator count."""
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin within 90s limit
    
    # Start with Batcher's odd-even mergesort (known correct)
    net = _batcher_odd_even_mergesort(N)
    
    if not _verify_network_fast(net, N):
        # Fallback: odd-even transposition
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Iteratively prune redundant comparators
    net = _try_prune(net, N, deadline)
    
    # Try additional pruning passes if time remains
    while time.time() < deadline - 10:
        new_net = _try_prune(net, N, deadline)
        if len(new_net) == len(net):
            break
        net = new_net
    
    return net
# EVOLVE-BLOCK-END
