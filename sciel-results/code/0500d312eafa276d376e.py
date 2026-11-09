import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 58.0  # 2s safety margin before 60s kill

    denoms = list(range(2, 49))
    n = len(denoms)
    
    # Calculate LCM of 2..48
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denoms)
    
    # Weights are L / d
    weights = [L // d for d in denoms]
    
    # Split into two halves
    # 23 and 24 is a good split for 47
    half = n // 2
    left_size = half
    right_size = n - half
    
    w_left = weights[:left_size]
    w_right = weights[left_size:]
    
    # Generate all valid subsets for the left half
    # We want sums < L
    left_sums = []
    # Use iterative expansion or recursion? 
    # 2^23 is about 8 million. 
    # Iterative:
    # sums = [0]
    # masks = [0]
    # for i, w in enumerate(w_left):
    #   new_sums = [s + w for s in sums if s + w < L]
    #   new_masks = [m | (1 << i) for m, s in zip(masks, sums) if s + w < L]
    #   sums.extend(new_sums)
    #   masks.extend(new_masks)
    
    # This is O(N * 2^N) which is too slow in pure Python for N=23.
    # 23 * 8M = 184M operations. Might be borderline or too slow.
    # Let's try to optimize or use a different approach.
    # Actually, 2^23 = 8,388,608.
    # A simple loop over all masks from 0 to 2^23 - 1 and computing sum?
    # That's 8M iterations. In Python, a tight loop of 8M is ~0.5-1s.
    # Computing sum for each mask naively is O(N) per mask, so 23 * 8M = 190M ops. Too slow.
    
    # Better: Dynamic programming / Iterative subset sum generation.
    # We can store (sum, mask) pairs.
    # Start with [(0, 0)]
    # For each weight w_i:
    #   For each existing (s, m):
    #      if s + w_i < L: add (s+w_i, m | (1<<i))
    
    # Let's implement this efficiently.
    
    def generate_subsets(weights_part, limit):
        sums = [0]
        masks = [0]
        for i, w in enumerate(weights_part):
            # We need to iterate over the CURRENT length of sums to avoid using newly added items in the same step
            current_len = len(sums)
            bit = 1 << i
            # Use local variables for speed
            s_list = sums
            m_list = masks
            new_s = []
            new_m = []
            
            # Optimization: if w is large, many won't fit.
            # Simple loop
            for j in range(current_len):
                s = s_list[j]
                ns = s + w
                if ns < limit:
                    new_s.append(ns)
                    new_m.append(m_list[j] | bit)
            
            if new_s:
                sums.extend(new_s)
                masks.extend(new_m)
                
        return sums, masks

    # Generate left subsets
    left_sums, left_masks = generate_subsets(w_left, L)
    
    # Generate right subsets
    right_sums, right_masks = generate_subsets(w_right, L)
    
    # Sort right subsets by sum
    # We want to find for each left sum, the largest right sum such that left + right < L
    # Sort right_sums and keep track of indices to recover masks
    # Pair sum and mask
    right_pairs = list(zip(right_sums, right_masks))
    right_pairs.sort(key=lambda x: x[0])
    
    # Extract sorted sums and masks for binary search
    r_sums = [p[0] for p in right_pairs]
    r_masks = [p[1] for p in right_pairs]
    
    import bisect
    
    best_sum = -1
    best_mask_left = 0
    best_mask_right = 0
    
    # Iterate over all left subsets
    # For each left sum, find max right sum < L - left_sum
    # Use binary search on r_sums
    
    for i in range(len(left_sums)):
        ls = left_sums[i]
        lm = left_masks[i]
        
        limit = L - ls
        
        # Find right index
        # We want the largest index j such that r_sums[j] < limit
        # bisect_left gives the first index where r_sums[idx] >= limit
        # So the valid index is idx - 1
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
                
                # Early termination? If best_sum is very close to L, we might stop.
                # But we want the exact best.
                # If we find L-1, we can break.
                if best_sum == L - 1:
                    break
    
    # Decode masks to indices
    indices = []
    
    # Left half corresponds to 0..left_size-1
    for i in range(left_size):
        if best_mask_left & (1 << i):
            indices.append(i)
            
    # Right half corresponds to left_size..n-1
    for i in range(right_size):
        if best_mask_right & (1 << i):
            indices.append(left_size + i)
            
    # Ensure non-empty
    if not indices:
        # Fallback: pick the smallest fraction (largest denominator)
        indices = [n - 1]
        
    return indices
