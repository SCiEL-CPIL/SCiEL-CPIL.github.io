"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import random

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

def _verify_network_fast(comps):
    """
    Verify sorting network using zero-one principle with early exit.
    Returns True if valid, False otherwise.
    """
    if not comps:
        return True
    
    x = _get_all_inputs().copy()
    for (i, j) in comps:
        a = x[:, i]
        b = x[:, j]
        x[:, i] = np.minimum(a, b)
        x[:, j] = np.maximum(a, b)
    
    # Check if sorted: x[:, k] <= x[:, k+1] for all k
    # Vectorized check
    diffs = np.diff(x, axis=1)
    if np.any(diffs < 0):
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

def _local_search(comps, deadline):
    """Local search to minimize comparator count while maintaining correctness."""
    best = list(comps)
    random.seed(42)
    
    # Strategy 1: Systematic removal passes
    # Multiple passes, each trying to remove one comparator at a time
    pass_num = 0
    while time.time() < deadline and pass_num < 20:
        pass_num += 1
        improved = False
        # Try each position, prefer later positions (more likely redundant)
        indices = list(range(len(best) - 1, -1, -1))
        random.shuffle(indices)  # Add randomness to break ties
        
        for idx in indices:
            if time.time() > deadline:
                break
            if len(best) <= 1:
                break
            # Try removing best[idx]
            trial = best[:idx] + best[idx+1:]
            if _verify_network_fast(trial):
                best = trial
                improved = True
                break  # Restart pass to catch new redundancies
    
    # Strategy 2: Random removals for polishing
    no_improve_count = 0
    max_no_improve = 10000
    
    while time.time() < deadline:
        if len(best) <= 1:
            break
        
        idx = random.randint(0, len(best) - 1)
        trial = best[:idx] + best[idx+1:]
        if _verify_network_fast(trial):
            best = trial
            no_improve_count = 0
        else:
            no_improve_count += 1
            
        if no_improve_count > max_no_improve:
            break
            
    return best

def construct():
    # Start with Batcher's odd-even merge sort
    comps = _batcher_oe_merge_sort(N)
    
    # Verify initial construction
    if not _verify_network_fast(comps):
        # Fallback to odd-even transposition (120 comparators)
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Local search to reduce comparators
    # Safety margin: 85 seconds (5s buffer for overhead)
    deadline = time.time() + 85.0
    best = _local_search(comps, deadline)
    
    return best
# EVOLVE-BLOCK-END
