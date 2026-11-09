import time
import bisect
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM
    L = 1
    for d in denominators:
        L = lcm(L, d)
    
    weights = [L // d for d in denominators]
    
    # Split into two halves
    half = n // 2
    left_size = half
    right_size = n - half
    
    left_weights = weights[:left_size]
    right_weights = weights[left_size:]
    
    # Function to generate all subset sums and masks
    def generate_sums(weights_list):
        size = len(weights_list)
        total_masks = 1 << size
        sums = [0] * total_masks
        # Standard DP to generate subset sums
        # Optimization: Use local variables for speed
        sums_local = sums
        for i in range(size):
            bit = 1 << i
            w = weights_list[i]
            # Iterate over the masks that don't have the i-th bit set
            # and set the i-th bit
            for mask in range(total_masks):
                if not (mask & bit):
                    sums_local[bit | mask] = sums_local[mask] + w
        
        # Filter out sums >= L and keep (sum, mask) pairs
        # Optimization: List comprehension is faster than append loop
        valid_pairs = [(sums[mask], mask) for mask in range(total_masks) if sums[mask] < L]
        
        return valid_pairs

    left_pairs = generate_sums(left_weights)
    
    if time.time() > deadline:
        return [0]
    
    right_pairs = generate_sums(right_weights)
    
    if time.time() > deadline:
        return [0]
    
    # Sort by sum
    left_pairs.sort(key=lambda x: x[0])
    right_pairs.sort(key=lambda x: x[0])
    
    if time.time() > deadline:
        return [0]
    
    # Extract just the sums for binary search
    right_sums = [rs for rs, _ in right_pairs]
    
    best_deficit = L
    best_left_mask = -1
    best_right_mask = -1
    
    # For each left sum, find the largest right sum such that left_sum + right_sum < L
    # Optimization: Pre-compute right_sums and right_pairs for faster access
    right_sums_local = right_sums
    right_pairs_local = right_pairs
    L_local = L
    bisect_left = bisect.bisect_left
    
    for ls, lm in left_pairs:
        if time.time() > deadline:
            break
        
        # We need: rs < L - ls
        target = L_local - ls
        # Find the rightmost index where right_sums[index] < target
        idx = bisect_left(right_sums_local, target) - 1
        
        if idx >= 0:
            rs, rm = right_pairs_local[idx]
            total = ls + rs
            if total < L_local:
                deficit = L_local - total
                if deficit < best_deficit:
                    best_deficit = deficit
                    best_left_mask = lm
                    best_right_mask = rm
                    if deficit == 1:
                        break
    
    # Fallback if no valid pair found
    if best_left_mask == -1:
        best_left_mask = 0
        best_right_mask = 0
        best_deficit = L

    # Reconstruct indices
    result_indices = []
    for k in range(left_size):
        if best_left_mask & (1 << k):
            result_indices.append(k)
            
    for k in range(right_size):
        if best_right_mask & (1 << k):
            result_indices.append(left_size + k)
            
    # Ensure non-empty
    if not result_indices:
        result_indices = [0]
        
    return result_indices
