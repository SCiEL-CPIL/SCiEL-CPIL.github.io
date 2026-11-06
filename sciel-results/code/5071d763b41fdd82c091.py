"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START

def _verify_network_fast(comparators):
    """Verify sorting network using zero-one principle with numpy.
    Returns True if the network sorts all 2^16 Boolean inputs."""
    if len(comparators) == 0:
        return N <= 1
    
    all_inputs = np.arange(65536, dtype=np.uint8)
    bits = ((all_inputs[:, None] >> np.arange(16)[None, :]) & 1).astype(np.int8)
    
    for (i, j) in comparators:
        mn = np.minimum(bits[:, i], bits[:, j])
        mx = np.maximum(bits[:, i], bits[:, j])
        bits[:, i] = mn
        bits[:, j] = mx
    
    diffs = np.diff(bits, axis=1)
    if np.any(diffs < 0):
        return False
    return True

def _odd_even_transposition(n):
    """Generate odd-even transposition sorting network for n wires."""
    return [(i, i + 1) for phase in range(n) for i in range(phase % 2, n - 1, 2)]

def _batcher_odd_even_merge(n):
    """Generate Batcher odd-even merge sort network for n wires (power of 2)."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    left = _batcher_odd_even_merge(half)
    right = [(i + half, j + half) for (i, j) in left]
    merge = _odd_even_merge(n)
    return left + right + merge

def _odd_even_merge(n):
    """Odd-even merge network for n wires (power of 2, n >= 2)."""
    if n == 2:
        return [(0, 1)]
    half = n // 2
    odd = _odd_even_merge(half)
    even = _odd_even_merge(half)
    odd_shifted = [(i + 1, j + 1) for (i, j) in odd]
    cross = [(i, i + 1) for i in range(0, n - 1, 2)]
    return odd + even + odd_shifted + cross

def _systematic_peel(network, deadline):
    """Systematically try to remove each comparator, keeping network valid.
    Returns the minimized network."""
    current = list(network)
    
    # Try removing from the end first (later comparators are often redundant)
    # Then try all positions
    for start_idx in range(len(current) - 1, -1, -1):
        if time.time() > deadline:
            break
        if start_idx >= len(current):
            continue
        candidate = list(current)
        candidate.pop(start_idx)
        if _verify_network_fast(candidate):
            current = candidate
    
    # Second pass: try removing from the beginning
    for start_idx in range(len(current)):
        if time.time() > deadline:
            break
        if start_idx >= len(current):
            break
        candidate = list(current)
        candidate.pop(start_idx)
        if _verify_network_fast(candidate):
            current = candidate
            start_idx -= 1  # Re-check same position
    
    # Third pass: try removing pairs of adjacent comparators
    i = 0
    while i < len(current) - 1:
        if time.time() > deadline:
            break
        candidate = list(current)
        candidate.pop(i)
        candidate.pop(i)  # Remove the next one (same index after first pop)
        if _verify_network_fast(candidate):
            current = candidate
            i = max(0, i - 2)
        else:
            i += 1
    
    return current

def _local_search_refine(network, deadline):
    """Local search to further minimize comparator count."""
    current = list(network)
    rng = random.Random(12345)
    
    while time.time() < deadline - 5.0:
        # Try removing a random comparator
        if len(current) <= 55:
            break
            
        candidate = list(current)
        idx = rng.randrange(len(current))
        candidate.pop(idx)
        
        if _verify_network_fast(candidate):
            current = candidate
        else:
            # Try swapping two comparators
            if len(current) >= 2:
                i, j = rng.sample(range(len(current)), 2)
                candidate = list(current)
                candidate[i], candidate[j] = candidate[j], candidate[i]
                if _verify_network_fast(candidate):
                    current = candidate
    
    return current

def _targeted_removal(network, deadline):
    """Try specific removal patterns based on network structure."""
    current = list(network)
    
    # Try removing comparators that are "far apart" in wire indices
    # (these are less likely to be critical for local sorting)
    for pass_num in range(3):
        if time.time() > deadline:
            break
        i = 0
        while i < len(current):
            if time.time() > deadline:
                break
            (wi, wj) = current[i]
            wire_dist = wj - wi
            # Only try removing if wire distance is large
            if wire_dist >= 8:
                candidate = list(current)
                candidate.pop(i)
                if _verify_network_fast(candidate):
                    current = candidate
                    continue
            i += 1
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill
    
    # Start with odd-even transposition (120 comparators, guaranteed valid)
    base = _odd_even_transposition(N)
    
    # Verify base
    if not _verify_network_fast(base):
        # Should not happen, but fallback
        return base
    
    # Phase 1: Systematic peeling (most effective)
    result = _systematic_peel(base, deadline - 30.0)
    
    # Phase 2: Targeted removal based on wire distance
    result = _targeted_removal(result, deadline - 15.0)
    
    # Phase 3: Local search refinement
    result = _local_search_refine(result, deadline)
    
    # Final verification
    if not _verify_network_fast(result):
        return base
    
    return result

# EVOLVE-BLOCK-END
