import time
from math import gcd
from functools import reduce
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin
    
    denoms = list(range(2, 49))
    n = len(denoms)
    
    # Calculate LCM of denominators
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denoms)
    
    # Weights are L / d
    weights = [L // d for d in denoms]
    
    # Split into two halves
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    def generate_subsets_numpy(weights_part):
        """Generate all subset sums using iterative doubling with NumPy."""
        sums = np.array([0], dtype=np.int64)
        masks = np.array([0], dtype=np.int64)
        
        for i, w in enumerate(weights_part):
            # Create new entries by adding w to all existing sums
            new_sums = sums + w
            new_masks = masks | (1 << i)
            
            # Filter: keep only sums < L
            valid = new_sums < L
            if np.any(valid):
                sums = np.concatenate([sums, new_sums[valid]])
                masks = np.concatenate([masks, new_masks[valid]])
            
            # Time check
            if time.time() > deadline:
                break
        
        return sums, masks
    
    # Generate for both halves
    sums1, masks1 = generate_subsets_numpy(w1)
    if time.time() > deadline:
        # Fallback: return a simple valid answer
        return [n - 1]
        
    sums2, masks2 = generate_subsets_numpy(w2)
    
    # Sort subsets2 by sum
    order2 = np.argsort(sums2)
    sums2_sorted = sums2[order2]
    masks2_sorted = masks2[order2]
    
    # For each sum in subset1, find the largest sum in subset2 such that sum1 + sum2 < L
    # This is equivalent to: sum2 < L - sum1
    # Use binary search (np.searchsorted) to find the rightmost valid index
    
    # We want the largest sum2 such that sum2 < L - sum1
    # np.searchsorted(sums2_sorted, L - sum1, side='left') gives the first index where sums2_sorted >= (L - sum1)
    # So we want index - 1 to get the largest sum2 < (L - sum1)
    
    # Vectorize: for each s1, compute target = L - s1, then find the rightmost index in sums2_sorted < target
    targets = L - sums1
    # Find the insertion point for each target (first index where sums2_sorted >= target)
    insert_points = np.searchsorted(sums2_sorted, targets, side='left')
    
    # The valid index is insert_point - 1 (the largest sum2 < target)
    valid_indices = insert_points - 1
    
    # Filter out invalid indices (where insert_point == 0, meaning no valid sum2)
    valid_mask = valid_indices >= 0
    
    if not np.any(valid_mask):
        # No valid combination found; return a simple valid answer
        return [n - 1]
    
    # For valid entries, compute total sum = sums1 + sums2_sorted[valid_indices]
    valid_indices_clipped = np.clip(valid_indices, 0, None)
    total_sums = sums1[valid_mask] + sums2_sorted[valid_indices_clipped[valid_mask]]
    
    # Find the maximum total sum (which minimizes the deficit 1 - sum)
    max_idx = np.argmax(total_sums)
    best_sum1 = sums1[valid_mask][max_idx]
    best_sum2 = sums2_sorted[valid_indices_clipped[valid_mask][max_idx]]
    
    # Get the corresponding masks
    best_mask1 = masks1[valid_mask][max_idx]
    best_mask2 = masks2_sorted[valid_indices_clipped[valid_mask][max_idx]]
    
    # Decode masks to indices
    indices = []
    
    # w1 corresponds to indices 0 to mid-1
    for i in range(mid):
        if best_mask1 & (1 << i):
            indices.append(i)
            
    # w2 corresponds to indices mid to n-1
    for i in range(len(w2)):
        if best_mask2 & (1 << i):
            indices.append(mid + i)
            
    # Ensure non-empty
    if not indices:
        # Fallback: pick the smallest fraction (largest denominator)
        indices = [n - 1]
        
    return indices

# EVOLVE-BLOCK-START
# (The function construct is defined above, but the prompt asks to keep the function name and signature.
# The current program has construct inside EVOLVE-BLOCK. I will place the new code inside the block.)
# EVOLVE-BLOCK-END
