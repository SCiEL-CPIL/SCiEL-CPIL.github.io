"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

N = 16

# EVOLVE-BLOCK-START

def _verify_network_fast(network, n=16):
    """Verify network sorts all 2^n inputs. Fails fast on invalid."""
    # We use a deterministic but shuffled order to fail fast on average
    # For reproducibility in the evaluator, we might want a fixed seed or just standard range
    # But standard range is slow for invalid networks if the counterexample is rare.
    # Let's stick to a simple range but with a trick:
    # Actually, for 16 wires, the counterexamples for "almost sorted" networks are often common.
    # Let's just use range for simplicity and determinism, but break early.
    
    for bits in range(1 << n):
        vals = [(bits >> i) & 1 for i in range(n)]
        for i, j in network:
            if vals[i] > vals[j]:
                vals[i], vals[j] = vals[j], vals[i]
        # Check if sorted
        seen_one = False
        for v in vals:
            if v == 1:
                seen_one = True
            elif seen_one:
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

def construct():
    """Return a sorting network for 16 wires."""
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin for safety
    
    # Start with Batcher's OMER (62 comparators for N=16)
    net = _batcher_odd_even_mergesort(16)
    
    # Greedy removal to minimize comparators
    i = 0
    while i < len(net) and time.time() < deadline:
        # Try to remove net[i]
        candidate = net[:i] + net[i+1:]
        
        # Verify candidate
        if _verify_network_fast(candidate, 16):
            # Removal successful, keep it removed
            net = candidate
            # Do not increment i, because the next element is now at index i
            # and we want to try removing it next
        else:
            # Removal failed, restore and move to next
            i += 1
            
    # Final verification (optional, but good for safety)
    # if time.time() < deadline and not _verify_network_fast(net, 16):
    #     # Should not happen, but fallback
    #     return _batcher_odd_even_mergesort(16)
    
    return net

# EVOLVE-BLOCK-END
