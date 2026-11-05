"""Low discrepancy on all arithmetic progressions - local search approach."""
# EVOLVE-BLOCK-START
import time
import random

def _max_ap_discrepancy(signs):
    """Compute max absolute signed sum over any arithmetic progression in [1..n].
    signs: list of +1/-1 indexed 0..n-1 (position i corresponds to index i+1).
    Returns the max absolute sum.
    """
    n = len(signs)
    max_abs = 0
    for d in range(1, n + 1):
        for r in range(d):
            # Build prefix sums for subsequence r, r+d, r+2d, ...
            prefix = [0]
            idx = r
            while idx < n:
                prefix.append(prefix[-1] + signs[idx])
                idx += d
            m = len(prefix)
            if m <= 1:
                continue
            min_so_far = prefix[0]
            max_so_far = prefix[0]
            for j in range(1, m):
                val = prefix[j]
                # Max subarray sum ending at j
                max_end = val - min_so_far
                if max_end > max_abs:
                    max_abs = max_end
                # Min subarray sum ending at j (we want its absolute value)
                min_end = val - max_so_far
                if -min_end > max_abs:
                    max_abs = -min_end
                if val < min_so_far:
                    min_so_far = val
                if val > max_so_far:
                    max_so_far = val
    return max_abs

def construct():
    deadline = time.time() + 85
    n = 256
    rng = random.Random(42)
    
    # Initialize with random signs
    best_signs = [rng.choice((-1, 1)) for _ in range(n)]
    best_val = _max_ap_discrepancy(best_signs)
    
    # Greedy local search: flip signs that reduce max AP discrepancy
    improved = True
    while improved and time.time() < deadline - 5:
        improved = False
        for i in range(n):
            old = best_signs[i]
            best_signs[i] = -old
            val = _max_ap_discrepancy(best_signs)
            if val < best_val:
                best_val = val
                improved = True
            else:
                best_signs[i] = old
    
    # Random restarts
    while time.time() < deadline - 10:
        signs = [rng.choice((-1, 1)) for _ in range(n)]
        val = _max_ap_discrepancy(signs)
        if val < best_val:
            best_signs = signs[:]
            best_val = val
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for i in range(n):
                old = signs[i]
                signs[i] = -old
                v = _max_ap_discrepancy(signs)
                if v < val:
                    val = v
                    improved = True
                    if val < best_val:
                        best_val = val
                        best_signs = signs[:]
                else:
                    signs[i] = old
    
    return best_signs
# EVOLVE-BLOCK-END
