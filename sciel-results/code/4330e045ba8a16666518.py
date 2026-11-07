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
    
    # Create array of (ratio, n) where ratio = n/sigma(n)
    # We want to find pairs with same sigma and ratio close to each other
    # Since a/b ≈ 3/2 and sigma(a)=sigma(b), we have a/sigma(a) ≈ (3/2)*b/sigma(b)
    
    # Build sorted list by ratio = n/sigma(n)
    # For each n, compute ratio as a float for sorting
    ratios = np.zeros(MAX_N + 1, dtype=np.float64)
    for n in range(1, MAX_N + 1):
        s = int(sigma[n])
        if s > 0:
            ratios[n] = n / s
    
    # Create indices sorted by ratio
    sorted_indices = np.argsort(ratios[1:], kind='mergesort') + 1
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        return [11, 6]
    
    # For each n in sorted order, look at neighbors with the same sigma value
    # We need to find pairs (a, b) where sigma(a) = sigma(b) and |a/b - 3/2| is minimized
    # This means |2a - 3b| / (2b) is minimized
    
    # Strategy: sort by ratio, then for each n, check nearby elements in the sorted list
    # that have the same sigma value
    
    # Build a dictionary mapping sigma value to list of n values (sorted by ratio)
    sigma_to_ns = {}
    for idx in sorted_indices:
        n = int(idx)
        s = int(sigma[n])
        if s not in sigma_to_ns:
            sigma_to_ns[s] = []
        sigma_to_ns[s].append(n)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 3:
        return [11, 6]
    
    # For each sigma group, find the pair that minimizes |a/b - 3/2|
    best_error = None
    best_pair = (11, 6)
    
    # Process groups sorted by size (smaller groups first for efficiency)
    sorted_groups = sorted(sigma_to_ns.items(), key=lambda x: len(x[1]))
    
    for s, nums in sorted_groups:
        if time.time() - start_time > DEADLINE - 2:
            break
        if len(nums) < 2:
            continue
        
        # nums is already sorted by ratio (since we built it from sorted_indices)
        # For each a in nums, find b in nums such that b ≈ 2a/3
        # Since nums is sorted by ratio = n/sigma(n), and all have same sigma,
        # ratio is proportional to n, so nums is also sorted by n
        
        # Binary search for target b = 2a/3
        for a in nums:
            if time.time() - start_time > DEADLINE - 1:
                break
            target = 2.0 * a / 3.0
            pos = bisect_left(nums, target)
            # Check pos and pos-1
            for b in (nums[pos], nums[pos - 1]):
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
