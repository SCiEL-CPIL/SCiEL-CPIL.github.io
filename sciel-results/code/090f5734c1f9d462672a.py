import numpy as np
import time
from fractions import Fraction

MAX_N = 10000000
DEADLINE = 85.0

def _compute_sigma(n_max):
    """Compute sigma(n) for all n in [1, n_max]."""
    sigma = np.ones(n_max + 1, dtype=np.int64)
    for i in range(2, n_max + 1):
        sigma[i::i] += i
    return sigma

def construct():
    start_time = time.time()
    
    # Compute sigma
    sigma = _compute_sigma(MAX_N)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        return [11, 6]
    
    # Group indices by sigma value
    # Use a dictionary mapping sigma value to list of indices
    groups = {}
    for n in range(1, MAX_N + 1):
        s = int(sigma[n])
        if s not in groups:
            groups[s] = []
        groups[s].append(n)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 3:
        return [11, 6]
    
    # Find best pair
    best_error = None
    best_pair = (11, 6)
    
    for s, nums in groups.items():
        if len(nums) < 2:
            continue
        if len(nums) > 1000:  # Skip very large groups for now, handle separately
            continue
        nums_sorted = sorted(nums)
        for i, a in enumerate(nums_sorted):
            # Want b ≈ 2a/3
            target = 2.0 * a / 3.0
            # Binary search for closest
            lo, hi = 0, len(nums_sorted) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if nums_sorted[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            # Check lo and lo-1
            candidates = []
            if lo < len(nums_sorted):
                candidates.append(nums_sorted[lo])
            if lo - 1 >= 0:
                candidates.append(nums_sorted[lo - 1])
            for b in candidates:
                if a == b:
                    continue
                # Compute |a/b - 3/2| = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                den = 2 * b
                if num == 0:
                    continue  # Invalid
                err = num / den
                if best_error is None or err < best_error:
                    best_error = err
                    best_pair = (a, b)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 1:
        return [int(best_pair[0]), int(best_pair[1])]
    
    return [int(best_pair[0]), int(best_pair[1])]
