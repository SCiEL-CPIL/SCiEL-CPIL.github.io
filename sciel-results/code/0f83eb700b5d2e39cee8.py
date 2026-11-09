"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 16

# EVOLVE-BLOCK-START
# Pre-compute all binary inputs once for fast verification
_all_inputs = None

def _get_all_inputs():
    """Get or create the pre-computed all binary inputs array."""
    global _all_inputs
    if _all_inputs is None:
        _all_inputs = np.zeros((65536, 16), dtype=np.int8)
        idx = np.arange(65536, dtype=np.int32)
        for i in range(16):
            _all_inputs[:, i] = (idx >> i) & 1
    return _all_inputs

def _verify_network(comps):
    """Verify sorting network using zero-one principle (all 2^N binary inputs)."""
    if not comps:
        return True
    
    x = _get_all_inputs().copy()
    for (i, j) in comps:
        a = x[:, i]
        b = x[:, j]
        x[:, i] = np.minimum(a, b)
        x[:, j] = np.maximum(a, b)
    
    diffs = np.diff(x, axis=1)
    return bool(np.all(diffs >= 0))

def _verify_network_fast(comps):
    """Faster verification using early termination on failure."""
    if not comps:
        return True
    
    x = _get_all_inputs().copy()
    for (i, j) in comps:
        a = x[:, i]
        b = x[:, j]
        x[:, i] = np.minimum(a, b)
        x[:, j] = np.maximum(a, b)
    
    # Check sortedness: each row should be non-decreasing
    # Equivalent: for all k, x[:, k] <= x[:, k+1]
    # We can check this more efficiently
    for k in range(15):
        if not np.all(x[:, k] <= x[:, k + 1]):
            return False
    return True

def _batcher_oe_merge_sort(n):
    """Build Batcher's odd-even merge sort network."""
    def oe_merge_network(n):
        """Odd-even merge network for 2n inputs (n from top half, n from bottom half)."""
        if n == 1:
            return [(0, 1)]
        # Compare (i, n+i) for i in 0..n-1
        comps = [(i, n + i) for i in range(n)]
        # Recursively merge even-indexed and odd-indexed subsequences
        # Even indices: 0, 2, 4, ... -> map to 0, 1, 2, ...
        # Odd indices: 1, 3, 5, ... -> map to 0, 1, 2, ...
        even_merge = oe_merge_network(n // 2)
        odd_merge = oe_merge_network(n // 2)
        # Apply even merge on even positions
        for (a, b) in even_merge:
            comps.append((2 * a, 2 * b))
        # Apply odd merge on odd positions
        for (a, b) in odd_merge:
            comps.append((2 * a + 1, 2 * b + 1))
        return comps
    
    def sort_network(n):
        """Sorting network using odd-even merge sort."""
        if n == 1:
            return []
        if n == 2:
            return [(0, 1)]
        if n % 2 == 1:
            # Handle odd number: sort n-1, then insert the last element
            base = sort_network(n - 1)
            # Insert element n-1 into sorted network of n-1 elements
            extra = [(i, n - 1) for i in range(n - 1)]
            return base + extra
        # Even: split into two halves
        half = n // 2
        left = sort_network(half)
        right = [(i + half, j + half) for (i, j) in sort_network(half)]
        # Odd-even merge the two sorted halves
        merge = oe_merge_network(half)
        return left + right + merge
    
    return sort_network(n)

def _local_search(comps, max_iters=50000, deadline=None):
    """Local search to minimize comparator count while maintaining correctness."""
    best = list(comps)
    best_count = len(best)
    
    start_time = time.time()
    iters = 0
    no_improve = 0
    
    import random
    random.seed(42)
    
    # Try removing comparators in a more systematic way
    # First, try removing from the end (later comparators are often more redundant)
    # Then try random positions
    
    # Strategy 1: Try removing from the end first
    for _ in range(min(20, len(best) - 1)):
        if deadline is not None and time.time() > deadline:
            break
        if len(best) <= 1:
            break
        trial = best[:-1]
        if _verify_network_fast(trial):
            best = trial
            best_count = len(best)
            no_improve = 0
            iters += 1
        else:
            no_improve += 1
            iters += 1
            if no_improve > 100:
                break
    
    # Strategy 2: Random removals
    while iters < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        
        if len(best) > 1:
            idx = random.randint(0, len(best) - 1)
            trial = best[:idx] + best[idx+1:]
            if _verify_network_fast(trial):
                best = trial
                best_count = len(best)
                no_improve = 0
            else:
                no_improve += 1
        
        iters += 1
        
        if no_improve > 3000:
            break
    
    return best

def construct():
    # Start with Batcher's odd-even merge sort
    comps = _batcher_oe_merge_sort(N)
    
    if not _verify_network(comps):
        # Fallback to odd-even transposition
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Local search to reduce comparators
    # Safety margin: use 75 seconds out of 90 second limit
    deadline = time.time() + 75.0
    best = _local_search(comps, max_iters=200000, deadline=deadline)
    
    return best
# EVOLVE-BLOCK-END
