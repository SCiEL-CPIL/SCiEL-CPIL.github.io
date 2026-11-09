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
    
    # Check if sorted: x[:, k] <= x[:, k+1] for all k
    # We can check this efficiently
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
    import random
    rng = random.Random(42)
    
    # Phase 1: Systematic back-to-front removal (multiple passes)
    while time.time() < deadline:
        improved = False
        for idx in range(len(best) - 1, -1, -1):
            if time.time() > deadline:
                break
            if len(best) <= 1:
                break
            trial = best[:idx] + best[idx+1:]
            if _verify_network(trial):
                best = trial
                improved = True
                break
        if not improved:
            break
    
    # Phase 2: Multi-pass random removal with increasing no-improve threshold
    no_improve_count = 0
    max_no_improve = 200
    for pass_num in range(100):
        if time.time() > deadline:
            break
        if len(best) <= 1:
            break
        
        improved_this_pass = False
        for _ in range(min(500, len(best) * 2)):
            if time.time() > deadline:
                break
            idx = rng.randint(0, len(best) - 1)
            trial = best[:idx] + best[idx+1:]
            if _verify_network(trial):
                best = trial
                improved_this_pass = True
                no_improve_count = 0
                break  # Restart this pass after a successful removal
        
        if not improved_this_pass:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                break
    
    # Phase 3: Targeted removal - try removing comparators that appear later
    # and have overlapping wires with earlier comparators
    if time.time() < deadline:
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for idx in range(len(best) - 1, -1, -1):
                if time.time() > deadline:
                    break
                if len(best) <= 1:
                    break
                # Check if this comparator's wires are "covered" by others
                i, j = best[idx]
                covered = False
                for k in range(idx):
                    a, b = best[k]
                    if (a == i or a == j) and (b == i or b == j):
                        covered = True
                        break
                if covered:
                    trial = best[:idx] + best[idx+1:]
                    if _verify_network(trial):
                        best = trial
                        improved = True
                        break
    
    return best

def construct():
    # Start with Batcher's odd-even merge sort
    comps = _batcher_oe_merge_sort(N)
    
    # Verify initial construction
    if not _verify_network(comps):
        # Fallback to odd-even transposition (120 comparators)
        return [(i, i + 1) for phase in range(N)
                for i in range(phase % 2, N - 1, 2)]
    
    # Local search to reduce comparators
    # Safety margin: 85 seconds (5s buffer for overhead)
    deadline = time.time() + 85.0
    best = _local_search(comps, deadline)
    
    # Final polish: one more systematic pass if time permits
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for idx in range(len(best) - 1, -1, -1):
                if time.time() > deadline - 2:
                    break
                if len(best) <= 1:
                    break
                trial = best[:idx] + best[idx+1:]
                if _verify_network(trial):
                    best = trial
                    improved = True
                    break
    
    return best
# EVOLVE-BLOCK-END
