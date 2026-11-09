"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 16

# EVOLVE-BLOCK-START
def _verify_network(comps):
    """Verify sorting network using zero-one principle (all 2^N binary inputs)."""
    if not comps:
        return True
    # Generate all 2^16 binary inputs as a (65536, 16) array
    # Using a fast method to create all bit patterns
    all_inputs = np.zeros((65536, 16), dtype=np.int8)
    for i in range(16):
        # For bit i, it is 1 for the upper half of each 2^(i+1) block
        # More simply: for each column i, the pattern is 0 for 2^i rows, 1 for 2^i rows, repeating
        # Actually, let's just use the format method but it's slow.
        # Let's use a vectorized approach.
        # The i-th bit of number k (0-indexed from LSB) is 1 if (k >> i) & 1 == 1
        # We can generate this with:
        # pattern = np.repeat(np.tile([0, 1], 2**(15-i)), 2**i)
        # But 2**(15-i) can be large.
        # Let's stick to the format method for correctness first, then optimize if needed.
        # Actually, 65536 * 16 is small enough that even a loop might be okay, but vectorized is better.
        # Let's try:
        # idx = np.arange(65536)
        # all_inputs[:, i] = (idx >> i) & 1
        pass
    
    idx = np.arange(65536, dtype=np.int32)
    for i in range(16):
        all_inputs[:, i] = (idx >> i) & 1
    
    x = all_inputs.copy()
    for (i, j) in comps:
        a = x[:, i]
        b = x[:, j]
        x[:, i] = np.minimum(a, b)
        x[:, j] = np.maximum(a, b)
    
    diffs = np.diff(x, axis=1)
    return bool(np.all(diffs >= 0))

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
    
    # Pre-compute some random indices to avoid overhead
    import random
    random.seed(42)
    
    while iters < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        
        # Try random removal of a comparator
        if len(best) > 1:
            idx = random.randint(0, len(best) - 1)
            trial = best[:idx] + best[idx+1:]
            if _verify_network(trial):
                best = trial
                best_count = len(best)
                no_improve = 0
            else:
                no_improve += 1
        
        iters += 1
        
        if no_improve > 5000:
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
    # Safety margin: assume we have ~85s left
    deadline = time.time() + 80.0
    best = _local_search(comps, max_iters=100000, deadline=deadline)
    
    return best
# EVOLVE-BLOCK-END
