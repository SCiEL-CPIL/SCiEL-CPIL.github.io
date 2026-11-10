import time
import math
import bisect
import random

def construct():
    deadline = time.time() + 55  # Safety margin
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of denominators
    def lcm(a, b):
        return a * b // math.gcd(a, b)
    
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Values scaled by L
    values = [L // d for d in denominators]
    
    # Split into two sets
    split = 24
    vals_1 = values[:split]
    vals_2 = values[split:]
    n1 = len(vals_1)
    n2 = len(vals_2)
    
    # Generate all subset sums for Set 1
    size_1 = 1 << n1
    sums_1 = [0] * size_1
    for mask in range(1, size_1):
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        sums_1[mask] = sums_1[prev_mask] + vals_1[idx]
        
    # Filter and sort Set 1 sums
    pairs_1 = []
    for mask in range(size_1):
        s = sums_1[mask]
        if s < L:
            pairs_1.append((s, mask))
            
    pairs_1.sort()
    
    # Pre-extract sums for faster binary search
    sums_1_sorted = [p[0] for p in pairs_1]
    masks_1_sorted = [p[1] for p in pairs_1]
    
    best_deficit = L
    best_mask_1 = 0
    best_mask_2 = 0
    best_sum_val = 0
    
    # Generate subset sums for Set 2 and search
    size_2 = 1 << n2
    sums_2 = [0] * size_2
    for mask in range(1, size_2):
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        prev_mask = mask ^ lsb
        sums_2[mask] = sums_2[prev_mask] + vals_2[idx]
        
    for mask_2 in range(size_2):
        if time.time() > deadline - 10:  # Leave time for local search
            break
            
        s2 = sums_2[mask_2]
        if s2 >= L:
            continue
            
        target = L - s2
        
        # Find largest s1 in sums_1_sorted such that s1 < target
        pos = bisect.bisect_left(sums_1_sorted, target)
        
        if pos > 0:
            s1 = sums_1_sorted[pos - 1]
            mask_1 = masks_1_sorted[pos - 1]
            total_sum = s1 + s2
            if total_sum < L:
                deficit = L - total_sum
                if deficit < best_deficit:
                    best_deficit = deficit
                    best_mask_1 = mask_1
                    best_mask_2 = mask_2
                    best_sum_val = total_sum
                    
    # Reconstruct indices
    indices = []
    for i in range(n1):
        if best_mask_1 & (1 << i):
            indices.append(i)
    for i in range(n2):
        if best_mask_2 & (1 << i):
            indices.append(split + i)
    
    # Ensure non-empty
    if not indices:
        indices = [0, 1, 5, 41]
    
    # Local search: try 2-swap moves to improve
    if time.time() < deadline - 5:
        current_indices = set(indices)
        current_sum = best_sum_val
        current_deficit = best_deficit
        
        # Precompute which indices are in/out
        in_set = [False] * n
        for idx in current_indices:
            in_set[idx] = True
        
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            
            # Try 2-swap moves: remove one in-set index, add one out-set index
            in_list = list(current_indices)
            out_list = [i for i in range(n) if not in_set[i]]
            
            for i in in_list:
                if time.time() > deadline - 5:
                    break
                for j in out_list:
                    # Check if swap improves deficit
                    new_sum = current_sum - values[i] + values[j]
                    if new_sum < L:
                        new_deficit = L - new_sum
                        if new_deficit < current_deficit:
                            # Accept the swap
                            current_indices.remove(i)
                            current_indices.add(j)
                            in_set[i] = False
                            in_set[j] = True
                            current_sum = new_sum
                            current_deficit = new_deficit
                            improved = True
                            break
                if time.time() > deadline - 5:
                    break
            
            # Also try single add (if we can add without exceeding L)
            if not improved:
                for j in out_list:
                    if time.time() > deadline - 5:
                        break
                    new_sum = current_sum + values[j]
                    if new_sum < L:
                        new_deficit = L - new_sum
                        if new_deficit < current_deficit:
                            current_indices.add(j)
                            in_set[j] = True
                            current_sum = new_sum
                            current_deficit = new_deficit
                            improved = True
                            break
            
            # Try single remove (if removing improves... actually removing increases deficit, so skip)
        
        if current_deficit < best_deficit:
            best_deficit = current_deficit
            best_sum_val = current_sum
            indices = list(current_indices)
    
    return indices
