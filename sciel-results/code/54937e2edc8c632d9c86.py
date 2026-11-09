"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 16

# EVOLVE-BLOCK-START
def _verify_network(comps):
    """Verify sorting network using zero-one principle (all 2^N binary inputs)."""
    if not comps:
        return True
    idx = np.arange(65536, dtype=np.int32)
    all_inputs = np.zeros((65536, 16), dtype=np.int8)
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
        even_merge = oe_merge_network(n // 2)
        odd_merge = oe_merge_network(n // 2)
        for (a, b) in even_merge:
            comps.append((2 * a, 2 * b))
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
            base = sort_network(n - 1)
            extra = [(i, n - 1) for i in range(n - 1)]
            return base + extra
        half = n // 2
        left = sort_network(half)
        right = [(i + half, j + half) for (i, j) in sort_network(half)]
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
    
    while iters < max_iters:
        if deadline is not None and time.time() > deadline:
            break
        
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
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Local search to reduce comparators
    deadline = time.time() + 85.0
    best = _local_search(comps, max_iters=100000, deadline=deadline)
    
    return best
# EVOLVE-BLOCK-END
