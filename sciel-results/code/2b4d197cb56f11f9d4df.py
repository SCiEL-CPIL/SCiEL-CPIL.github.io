import time
from math import gcd
from functools import reduce
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # 2s safety margin before 60s kill

    denoms = list(range(2, 49))
    n = len(denoms)
    
    # Calculate LCM of 2..48
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denoms)
    
    # Create list of (weight, original_index)
    # Weights are L / d
    items = [(L // d, i) for i, d in enumerate(denoms)]
    
    # Sort by weight descending to improve pruning in subset generation
    items.sort(key=lambda x: x[0], reverse=True)
    
    sorted_weights = [w for w, i in items]
    
    # Split into two halves
    half = n // 2
    left_size = half
    right_size = n - half
    
    w_left = sorted_weights[:left_size]
    w_right = sorted_weights[left_size:]
    
    def generate_subsets(weights_part, limit):
        sums = [0]
        masks = [0]
        for i, w in enumerate(weights_part):
            current_len = len(sums)
            bit = 1 << i
            new_s = []
            new_m = []
            append_s = new_s.append
            append_m = new_m.append
            
            s_list = sums
            m_list = masks
            
            for j in range(current_len):
                s = s_list[j]
                ns = s + w
                if ns < limit:
                    append_s(ns)
                    append_m(m_list[j] | bit)
            
            sums.extend(new_s)
            masks.extend(new_m)
                
        return sums, masks

    # Generate left subsets
    left_sums, left_masks = generate_subsets(w_left, L)
    
    # Generate right subsets
    right_sums, right_masks = generate_subsets(w_right, L)
    
    # Sort right subsets by sum
    right_pairs = list(zip(right_sums, right_masks))
    right_pairs.sort(key=lambda x: x[0])
    
    r_sums = [p[0] for p in right_pairs]
    r_masks = [p[1] for p in right_pairs]
    
    best_sum = -1
    best_mask_left = 0
    best_mask_right = 0
    
    # Iterate over all left subsets
    for i in range(len(left_sums)):
        ls = left_sums[i]
        lm = left_masks[i]
        
        limit = L - ls
        
        idx = bisect.bisect_left(r_sums, limit)
        
        if idx > 0:
            j = idx - 1
            rs = r_sums[j]
            rm = r_masks[j]
            
            total = ls + rs
            if total > best_sum:
                best_sum = total
                best_mask_left = lm
                best_mask_right = rm
                
                if best_sum == L - 1:
                    break
    
    # Decode masks to indices
    # The masks correspond to the sorted order.
    # We need to map bit position in the mask to the original index.
    # Left mask bits 0..left_size-1 correspond to items[0]..items[left_size-1]
    # Right mask bits 0..right_size-1 correspond to items[left_size]..items[n-1]
    
    indices = []
    
    for i in range(left_size):
        if best_mask_left & (1 << i):
            # items[i] is (weight, original_index)
            indices.append(items[i][1])
            
    for i in range(right_size):
        if best_mask_right & (1 << i):
            # items[left_size + i] is (weight, original_index)
            indices.append(items[left_size + i][1])
            
    # Ensure non-empty
    if not indices:
        indices = [n - 1]
        
    return indices
