"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 16

# EVOLVE-BLOCK-START
import time
import random

def _verify_network_fast(network, n=16):
    """Verify network sorts all 2^n inputs using zero-one principle with early exit."""
    for mask in range(1 << n):
        vals = [(mask >> i) & 1 for i in range(n)]
        for i, j in network:
            if vals[i] > vals[j]:
                vals[i], vals[j] = vals[j], vals[i]
        # Check if sorted: vals should be non-decreasing
        for k in range(n - 1):
            if vals[k] > vals[k + 1]:
                return False
    return True

def _verify_network_sampled(network, n=16, samples=1000, rng=None):
    """Fast approximate verification using random sampling for local search."""
    if rng is None:
        rng = random.Random()
    for _ in range(samples):
        mask = rng.getrandbits(n)
        vals = [(mask >> i) & 1 for i in range(n)]
        for i, j in network:
            if vals[i] > vals[j]:
                vals[i], vals[j] = vals[j], vals[i]
        for k in range(n - 1):
            if vals[k] > vals[k + 1]:
                return False
    return True

def _batcher_odd_even_mergesort(n):
    """Generate Batcher's odd-even mergesort network for n wires.
    Returns list of (i, j) comparator pairs."""
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

def _local_search_prune(network, n=16, deadline=None, rng=None, sample_size=500):
    """Iteratively try to remove redundant comparators from a correct network.
    Returns the pruned network (still correct)."""
    if rng is None:
        rng = random.Random(42)
    
    best = list(network)
    best_len = len(best)
    
    while True:
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        # Try removing each comparator in random order
        indices = list(range(len(best)))
        rng.shuffle(indices)
        
        improved = False
        for idx in indices:
            if deadline is not None and time.time() > deadline - 5.0:
                break
            
            candidate = best[:idx] + best[idx+1:]
            
            # Quick sampled check first
            if not _verify_network_sampled(candidate, n, samples=sample_size, rng=rng):
                continue
            
            # Full verification
            if _verify_network_fast(candidate, n):
                best = candidate
                best_len -= 1
                improved = True
                break  # Restart with new best
        
        if not improved:
            break
    
    return best

def construct():
    """Return a sorting network for 16 wires."""
    start_time = time.time()
    deadline = start_time + 80.0  # Leave 10s safety margin
    
    # Start from Batcher's odd-even mergesort
    net = _batcher_odd_even_mergesort(16)
    
    # Verify initial network
    if not _verify_network_fast(net, 16):
        # Fallback to odd-even transposition
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Attempt local search pruning
    rng = random.Random(12345)
    pruned = _local_search_prune(net, 16, deadline=deadline, rng=rng)
    
    # Final verification
    if _verify_network_fast(pruned, 16):
        return pruned
    
    # If pruning somehow broke it (shouldn't happen), return original
    return net
# EVOLVE-BLOCK-END
