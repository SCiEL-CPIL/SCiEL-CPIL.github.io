import time
import bisect
from math import gcd
from functools import reduce
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denominators)
    weights = [L // d for d in denominators]
    
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    # Generate sums for first half using NumPy
    size1 = 1 << mid
    # Initialize
    sums1 = np.zeros(size1, dtype=object) # Object to handle big ints? Or int64?
    # L is ~ 5e17. int64 max is 9e18. So int64 is safe.
    sums1 = np.zeros(size1, dtype=np.int64)
    masks1 = np.zeros(size1, dtype=np.int64)
    
    current_size = 1
    for i, w in enumerate(w1):
        # sums1[current_size : current_size*2] = sums1[:current_size] + w
        sums1[current_size:current_size*2] = sums1[:current_size] + w
        masks1[current_size:current_size*2] = masks1[:current_size] | (1 << i)
        current_size *= 2
        
        if time.time() > deadline:
            break

    # Sort
    order = np.argsort(sums1)
    sorted_sums = sums1[order]
    sorted_masks = masks1[order]
    
    # Convert to Python lists for bisect? Or use numpy searchsorted?
    # numpy.searchsorted is fast.
    
    best_deficit = None
    best_mask1 = 0
    best_mask2 = 0
    
    # Generate sums for second half
    size2 = 1 << (n - mid)
    sums2 = np.zeros(size2, dtype=np.int64)
    masks2 = np.zeros(size2, dtype=np.int64)
    
    current_size = 1
    for i, w in enumerate(w2):
        sums2[current_size:current_size*2] = sums2[:current_size] + w
        masks2[current_size:current_size*2] = masks2[:current_size] | (1 << i)
        current_size *= 2
        
        if time.time() > deadline:
            break

    # Iterate over sums2
    # We can vectorize the search?
    # For each s2, target = L - s2 - 1
    # idx = searchsorted(sorted_sums, target, side='right') - 1
    
    targets = L - sums2 - 1
    # Filter valid targets (>= 0)
    valid = targets >= 0
    targets_valid = targets[valid]
    s2_valid = sums2[valid]
    m2_valid = masks2[valid]
    
    if len(targets_valid) == 0:
        # Fallback
        return []

    # searchsorted
    # Returns index of where to insert to keep order
    # side='right' means index after any existing entries
    # We want the largest s1 <= target.
    # So idx = searchsorted(sorted_sums, target, side='right') - 1
    
    idxs = np.searchsorted(sorted_sums, targets_valid, side='right') - 1
    
    # Filter valid idxs (>= 0)
    valid_idx = idxs >= 0
    if not np.any(valid_idx):
        return []
        
    idxs_valid = idxs[valid_idx]
    s1_candidates = sorted_sums[idxs_valid]
    m1_candidates = sorted_masks[idxs_valid]
    
    s2_candidates = s2_valid[valid_idx]
    m2_candidates = m2_valid[valid_idx]
    
    total_sums = s1_candidates + s2_candidates
    deficits = L - total_sums
    
    # Find min deficit
    min_def_idx = np.argmin(deficits)
    best_deficit = deficits[min_def_idx]
    best_mask1 = int(m1_candidates[min_def_idx])
    best_mask2 = int(m2_candidates[min_def_idx])
    
    if time.time() > deadline:
        # If we are close to deadline, return best found so far?
        # The vectorized operation is fast, so we should be done.
        pass

    # Reconstruct indices
    indices = []
    for i in range(mid):
        if best_mask1 & (1 << i):
            indices.append(i)
    for i in range(len(w2)):
        if best_mask2 & (1 << i):
            indices.append(mid + i)
            
    return indices
