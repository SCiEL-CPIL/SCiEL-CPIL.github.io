"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

N = 16

# EVOLVE-BLOCK-START

def _batcher_oes(n):
    """Batcher's odd-even merge sort network for n elements (n must be power of 2).
    Returns list of (i, j) comparators where min goes to wire i, max to wire j."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    # Sort first half and second half recursively
    left = _batcher_oes(half)
    right = [(i + half, j + half) for i, j in left]
    
    # Odd-even merge the two halves
    merge = _odd_even_merge(n)
    
    return left + right + merge


def _odd_even_merge(n):
    """Odd-even merge of two sorted sequences of length n/2 each.
    Input: positions 0..n/2-1 are first sorted sequence, n/2..n-1 are second.
    Returns comparators to merge them into one sorted sequence of length n."""
    if n <= 1:
        return []
    if n == 2:
        return [(0, 1)]
    
    half = n // 2
    quarter = half // 2
    
    # Merge odd-indexed elements and even-indexed elements recursively
    # Odd positions in first half: 1, 3, 5, ... (0-indexed within first half)
    # Odd positions in second half: n/2+1, n/2+3, ...
    # Even positions in first half: 0, 2, 4, ...
    # Even positions in second half: n/2, n/2+2, ...
    
    # For odd-even merge, we compare:
    # (0, n/2), (1, n/2+1), (2, n/2+2), ... for the "cross" comparisons
    # Then recursively merge odd and even subsequences
    
    result = []
    
    # Cross comparisons: compare element i with element i + n/2
    for i in range(half):
        result.append((i, i + half))
    
    # Now we need to recursively merge the odd and even subsequences
    # Odd subsequence: positions 1, 3, 5, ... in the merged result
    # Even subsequence: positions 0, 2, 4, ... in the merged result
    
    # The odd elements after cross-compare are at positions 1, 3, 5, ..., n-1
    # The even elements after cross-compare are at positions 0, 2, 4, ..., n-2
    
    # Recursively merge odd elements (positions 1,3,5,...)
    # These correspond to a smaller odd-even merge problem
    # Map: odd position 2k+1 -> index k in the sub-problem
    odd_merge = _odd_even_merge(half)
    for i, j in odd_merge:
        result.append((2*i + 1, 2*j + 1))
    
    # Recursively merge even elements (positions 0,2,4,...)
    even_merge = _odd_even_merge(half)
    for i, j in even_merge:
        result.append((2*i, 2*j))
    
    return result


def _verify_network_vectorized(comparators, n=N):
    """Verify network using zero-one principle with numpy vectorization."""
    # Generate all 2^n inputs as a 2D array
    masks = np.arange(1 << n, dtype=np.uint32)
    # Create input matrix: shape (2^n, n)
    # wires[k, i] = bit i of mask k
    wires = np.zeros((1 << n, n), dtype=np.uint8)
    for i in range(n):
        wires[:, i] = (masks >> i) & 1
    
    # Apply each comparator
    for i, j in comparators:
        # If wires[:, i] > wires[:, j], swap
        swap_mask = wires[:, i] > wires[:, j]
        if np.any(swap_mask):
            wires[swap_mask, i] = 1
            wires[swap_mask, j] = 0
    
    # Check if all rows are sorted (non-decreasing)
    # A row is sorted if wires[k, i] <= wires[k, i+1] for all i
    diffs = wires[:, 1:] - wires[:, :-1]
    if np.any(diffs < 0):
        return False
    return True


def _verify_network_fast_scalar(comparators, n=N):
    """Verify network using zero-one principle, scalar version for small networks."""
    for mask in range(1 << n):
        wires = [(mask >> i) & 1 for i in range(n)]
        for i, j in comparators:
            if wires[i] > wires[j]:
                wires[i], wires[j] = wires[j], wires[i]
        # Check if sorted
        for k in range(n - 1):
            if wires[k] > wires[k + 1]:
                return False
    return True


def _local_search(comparators, max_time=60.0, n=N):
    """Local search to reduce comparator count by removing redundant comparators."""
    start_time = time.time()
    best = list(comparators)
    best_len = len(best)
    rng = random.Random(42)
    
    # Precompute for fast verification
    def verify(comps):
        return _verify_network_vectorized(comps, n)
    
    # First, try removing each comparator in order (greedy)
    i = 0
    while i < len(best):
        if time.time() - start_time > max_time - 5.0:
            break
        candidate = best[:i] + best[i+1:]
        if verify(candidate):
            best = candidate
            best_len -= 1
            # Don't increment i, check same position again
        else:
            i += 1
    
    # Then, try random removals and swaps
    while time.time() - start_time < max_time - 5.0:
        # Try removing a random comparator
        idx = rng.randint(0, len(best) - 1)
        candidate = best[:idx] + best[idx+1:]
        if verify(candidate):
            best = candidate
            best_len -= 1
            continue
        
        # Try swapping two comparators (reordering)
        if len(best) >= 2:
            i, j = rng.sample(range(len(best)), 2)
            candidate = best[:]
            candidate[i], candidate[j] = candidate[j], candidate[i]
            if verify(candidate):
                best = candidate
        
        # Try replacing a comparator with a different wire pair
        if len(best) > 0:
            idx = rng.randint(0, len(best) - 1)
            new_i = rng.randint(0, n-1)
            new_j = rng.randint(0, n-1)
            if new_i < new_j:
                candidate = best[:]
                candidate[idx] = (new_i, new_j)
                if verify(candidate):
                    best = candidate
    
    return best


def construct():
    """Construct a sorting network for 16 wires."""
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    # Start with Batcher's odd-even merge sort network for 16 elements
    # This should be around 60-70 comparators
    base = _batcher_oes(N)
    
    # Verify it's correct
    if not _verify_network_fast_scalar(base):
        # Fallback to odd-even transposition (known correct, 120 comparators)
        base = [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Local search to improve
    if time.time() - start_time < deadline - 10.0:
        improved = _local_search(base, max_time=deadline - time.time() - 5.0, n=N)
        if len(improved) < len(base):
            base = improved
    
    return base

# EVOLVE-BLOCK-END
