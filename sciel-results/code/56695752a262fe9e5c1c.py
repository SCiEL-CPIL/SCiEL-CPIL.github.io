"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START

def _verify_network_fast(comparators):
    """Verify sorting network using zero-one principle with numpy.
    Returns True if the network sorts all 2^16 Boolean inputs."""
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

def _batcher_oes(n):
    """Batcher odd-even merge sort for n wires (power of 2)."""
    if n == 1:
        return []
    if n == 2:
        return [(0, 1)]
    half = n // 2
    left = _batcher_oes(half)
    right = [(i + half, j + half) for (i, j) in left]
    merge = _odd_even_merge(n)
    return left + right + merge

def _odd_even_merge(n):
    """Odd-even merge for n wires (power of 2, n >= 2)."""
    if n == 2:
        return [(0, 1)]
    half = n // 2
    odd = _odd_even_merge(half)
    even = _odd_even_merge(half)
    odd_shifted = [(i + 1, j + 1) for (i, j) in odd]
    cross = [(i, i + 1) for i in range(0, n - 1, 2)]
    return odd + even + odd_shifted + cross

def _build_butterfly_network():
    """Build a butterfly-based sorting network for 16 wires.
    
    Uses butterfly layers with additional comparators to ensure sorting.
    The butterfly has log2(16)=4 stages, each with 8 comparators = 32 comparators.
    We add extra layers to ensure full sorting.
    """
    comparators = []
    n = 16
    logn = 4
    
    # Butterfly layers
    for stage in range(logn):
        for i in range(n):
            j = i ^ (1 << stage)
            if i < j:
                comparators.append((i, j))
    
    # Add adjacent comparators in multiple passes to ensure sorting
    # This is similar to odd-even transposition but interleaved with butterfly
    for pass_num in range(4):
        for i in range(0, n - 1, 2):
            comparators.append((i, i + 1))
        for i in range(1, n - 1, 2):
            comparators.append((i, i + 1))
    
    # Additional butterfly layers for robustness
    for stage in range(logn):
        for i in range(n):
            j = i ^ (1 << stage)
            if i < j:
                comparators.append((i, j))
    
    # Final adjacent passes
    for pass_num in range(2):
        for i in range(0, n - 1, 2):
            comparators.append((i, i + 1))
        for i in range(1, n - 1, 2):
            comparators.append((i, i + 1))
    
    return comparators

def _systematic_removal(network, time_budget=50.0):
    """Systematically try to remove each comparator one at a time.
    This is more effective than random removal for finding minimal networks.
    """
    deadline = time.time() + time_budget
    current = list(network)
    
    if not _verify_network_fast(current):
        return current
    
    # Keep trying to remove comparators until we can't anymore
    progress = True
    while progress and time.time() < deadline - 5.0:
        progress = False
        # Try removing each comparator in order
        for idx in range(len(current)):
            if time.time() >= deadline - 5.0:
                break
            candidate = current[:idx] + current[idx+1:]
            if _verify_network_fast(candidate):
                current = candidate
                progress = True
                break  # Restart from beginning after successful removal
    
    return current

def _greedy_minimize(network, time_budget=50.0):
    """Greedy minimization: repeatedly find and remove any removable comparator.
    Uses a smarter ordering to find removable comparators faster.
    """
    deadline = time.time() + time_budget
    current = list(network)
    
    if not _verify_network_fast(current):
        return current
    
    # Try to remove comparators, prioritizing those that appear less critical
    # First, count how many times each wire pair appears
    pair_count = {}
    for (i, j) in current:
        pair_count[(i, j)] = pair_count.get((i, j), 0) + 1
    
    # Sort indices by pair frequency (try removing frequent ones first? or rare ones?)
    # Try removing each comparator, keeping the one that allows removal
    removed_any = True
    while removed_any and time.time() < deadline - 5.0:
        removed_any = False
        # Try each position
        for idx in range(len(current)):
            if time.time() >= deadline - 5.0:
                break
            candidate = current[:idx] + current[idx+1:]
            if _verify_network_fast(candidate):
                current = candidate
                removed_any = True
                break
    
    return current

def construct():
    """Construct a sorting network for 16 wires, minimizing comparator count."""
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    # Start with Batcher OES (known correct, ~63 comparators)
    base = _batcher_oes(16)
    
    if not _verify_network_fast(base):
        # Fallback to odd-even transposition (120 comparators, definitely correct)
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    best = base
    best_count = len(base)
    
    # Try systematic removal on Batcher
    if time.time() < deadline - 15.0:
        result = _systematic_removal(base, time_budget=min(30.0, deadline - time.time() - 10.0))
        if _verify_network_fast(result) and len(result) < best_count:
            best = result
            best_count = len(result)
    
    # Try butterfly-based approach
    if time.time() < deadline - 10.0:
        butterfly = _build_butterfly_network()
        if _verify_network_fast(butterfly) and len(butterfly) < best_count:
            best = butterfly
            best_count = len(butterfly)
            # Try to minimize the butterfly network
            if time.time() < deadline - 10.0:
                result = _systematic_removal(butterfly, time_budget=min(20.0, deadline - time.time() - 10.0))
                if _verify_network_fast(result) and len(result) < best_count:
                    best = result
                    best_count = len(result)
    
    # Additional local search from best found
    if time.time() < deadline - 10.0 and best_count > 55:
        rng = random.Random(42)
        current = list(best)
        while time.time() < deadline - 5.0:
            # Try random removal
            if len(current) > 50:
                idx = rng.randrange(len(current))
                candidate = current[:idx] + current[idx+1:]
                if _verify_network_fast(candidate):
                    current = candidate
                    best = current
                    best_count = len(current)
                else:
                    # Try swapping two comparators
                    if len(current) >= 2:
                        i, j = rng.sample(range(len(current)), 2)
                        candidate2 = list(current)
                        candidate2[i], candidate2[j] = candidate2[j], candidate2[i]
                        if _verify_network_fast(candidate2):
                            current = candidate2
                            best = current
                            best_count = len(current)
    
    # Final verification
    if not _verify_network_fast(best):
        return base
    
    return best

# EVOLVE-BLOCK-END
