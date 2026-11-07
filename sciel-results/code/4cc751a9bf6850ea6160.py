import numpy as np
import time
from fractions import Fraction
from bisect import bisect_left

MAX_N = 10000000
DEADLINE = 85.0

def _compute_sigma(n_max):
    """Compute sigma(n) for all n in [1, n_max] using vectorized approach."""
    sigma = np.zeros(n_max + 1, dtype=np.int64)
    for i in range(1, n_max + 1):
        sigma[i::i] += i
    return sigma

def construct():
    start_time = time.time()
    
    # Compute sigma
    sigma = _compute_sigma(MAX_N)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 10:
        return [11, 6]
    
    # Group indices by sigma value
    groups = {}
    for n in range(1, MAX_N + 1):
        s = int(sigma[n])
        if s not in groups:
            groups[s] = []
        groups[s].append(n)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        return [11, 6]
    
    # Find best pair
    best_error = None
    best_pair = (11, 6)
    
    # Sort groups by size, process smaller groups first
    sorted_groups = sorted(groups.items(), key=lambda x: len(x[1]))
    
    for s, nums in sorted_groups:
        if time.time() - start_time > DEADLINE - 2:
            break
        if len(nums) < 2:
            continue
        nums_sorted = sorted(nums)
        for i, a in enumerate(nums_sorted):
            if time.time() - start_time > DEADLINE - 1:
                break
            # Want b ≈ 2a/3
            target = 2.0 * a / 3.0
            # Binary search for closest
            pos = bisect_left(nums_sorted, target)
            # Check pos and pos-1
            for b in (nums_sorted[pos], nums_sorted[pos - 1]):
                if b == a:
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
    
    return [int(best_pair[0]), int(best_pair[1])]
