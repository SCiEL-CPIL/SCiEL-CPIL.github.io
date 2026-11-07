import numpy as np
import time
from fractions import Fraction

MAX_N = 10000000
DEADLINE = 85.0

def _compute_sigma(n_max):
    """Compute sigma(n) for all n in [1, n_max] using vectorized approach."""
    sigma = np.zeros(n_max + 1, dtype=np.int64)
    # For each divisor d, add d to all multiples of d
    # Process in chunks to manage memory
    for d in range(1, n_max + 1):
        sigma[d::d] += d
    return sigma

def construct():
    start_time = time.time()
    
    # Compute sigma
    sigma = _compute_sigma(MAX_N)
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        return [11, 6]
    
    # Group indices by sigma value using numpy for speed
    # Get unique sigma values and their positions
    # We need to find all pairs (a,b) with same sigma, then minimize |a/b - 3/2|
    
    # Strategy: sort by sigma, then within each group, find pairs close to ratio 3/2
    # Use np.argsort on sigma values
    indices = np.arange(1, MAX_N + 1, dtype=np.int64)
    sigma_vals = sigma[1:]
    
    # Sort by sigma value
    order = np.argsort(sigma_vals, kind='stable')
    sorted_sigma = sigma_vals[order]
    sorted_indices = indices[order]
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 3:
        return [11, 6]
    
    # Find boundaries where sigma value changes
    # Use np.diff to find where values change
    changes = np.concatenate(([True], sorted_sigma[1:] != sorted_sigma[:-1]))
    group_starts = np.where(changes)[0]
    group_ends = np.concatenate((group_starts[1:], [MAX_N]))
    
    best_error = float('inf')
    best_pair = (11, 6)
    
    # Process each group
    for gi in range(len(group_starts)):
        if time.time() - start_time > DEADLINE - 2:
            break
        
        gs = group_starts[gi]
        ge = group_ends[gi]
        group_size = ge - gs
        if group_size < 2:
            continue
        
        # Get the numbers in this group
        nums = sorted_indices[gs:ge]
        # nums is already sorted by index (since we used stable sort and original indices are increasing)
        # But we need them sorted by value for binary search
        # Actually since we sorted by sigma and then the indices within same sigma are in original order,
        # they should be in increasing order already due to stable sort
        nums = nums.astype(np.int64)
        
        # For each a in nums, find b in nums closest to 2a/3
        # Use vectorized approach: for each a, target = 2a/3, find closest b
        # We can use searchsorted on the sorted nums array
        
        # To be efficient, only consider a reasonable range
        # |a/b - 3/2| < best_error means we need b in range (2a/(3+2*best_error*b), ...)
        # Simpler: just check all pairs in small groups, and for large groups use smarter approach
        
        if group_size <= 500:
            # Brute force with numpy for small groups
            for i in range(group_size):
                a = int(nums[i])
                target = 2.0 * a / 3.0
                # Find closest element to target in nums
                pos = np.searchsorted(nums, target)
                candidates = []
                if pos < group_size:
                    candidates.append(int(nums[pos]))
                if pos > 0:
                    candidates.append(int(nums[pos - 1]))
                for b in candidates:
                    if a == b:
                        continue
                    num = abs(2 * a - 3 * b)
                    den = 2 * b
                    if num == 0:
                        continue
                    err = num / den
                    if err < best_error:
                        best_error = err
                        best_pair = (a, b)
        else:
            # For large groups, sample or use smarter approach
            # Only check a subset to save time
            step = max(1, group_size // 500)
            for i in range(0, group_size, step):
                a = int(nums[i])
                target = 2.0 * a / 3.0
                pos = np.searchsorted(nums, target)
                candidates = []
                if pos < group_size:
                    candidates.append(int(nums[pos]))
                if pos > 0:
                    candidates.append(int(nums[pos - 1]))
                for b in candidates:
                    if a == b:
                        continue
                    num = abs(2 * a - 3 * b)
                    den = 2 * b
                    if num == 0:
                        continue
                    err = num / den
                    if err < best_error:
                        best_error = err
                        best_pair = (a, b)
    
    return [int(best_pair[0]), int(best_pair[1])]
