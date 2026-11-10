import time
from math import gcd
from bisect import bisect_left

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # 1. Calculate LCM of 2..48
    L = 1
    for i in range(2, 49):
        L = L * i // gcd(L, i)
        
    # 2. Weights
    # Indices 0..46 correspond to denominators 2..48
    weights = [L // (i + 2) for i in range(47)]
    
    total_weight = sum(weights)
    target = total_weight - L
    
    # We want subset R such that sum(R) > target, minimizing sum(R).
    
    # Split
    mid = 23
    weights_A = weights[:mid]
    weights_B = weights[mid:]
    
    def gen_sums_masks(weights_list):
        n = len(weights_list)
        # Use separate arrays for sums and masks to reduce overhead
        sums = [0]
        masks = [0]
        for k in range(n):
            w = weights_list[k]
            bit = 1 << k
            current_len = len(sums)
            for i in range(current_len):
                sums.append(sums[i] + w)
                masks.append(masks[i] | bit)
        return sums, masks

    # Generate sums for A
    sums_A, masks_A = gen_sums_masks(weights_A)
    if time.time() > deadline:
        return [0] # Fallback
        
    # Generate sums for B
    sums_B, masks_B = gen_sums_masks(weights_B)
    if time.time() > deadline:
        return [0] # Fallback

    # Sort B by sum, keeping masks aligned
    b_pairs = list(zip(sums_B, masks_B))
    b_pairs.sort(key=lambda x: x[0])
    sums_B = [p[0] for p in b_pairs]
    masks_B = [p[1] for p in b_pairs]
    
    best_delta = None
    best_mask_A = 0
    best_mask_B = 0
    
    # Iterate through A
    for i in range(len(sums_A)):
        s_A = sums_A[i]
        m_A = masks_A[i]
        
        req = target - s_A + 1
        if req < 0:
            req = 0
            
        # Find first index in sums_B >= req
        idx = bisect_left(sums_B, req)
        
        if idx < len(sums_B):
            s_B = sums_B[idx]
            total_R = s_A + s_B
            if total_R > target:
                delta = total_R - target
                if best_delta is None or delta < best_delta:
                    best_delta = delta
                    best_mask_A = m_A
                    best_mask_B = masks_B[idx]
                    
                    if best_delta == 1:
                        break
                        
    # Construct indices for S (complement of R)
    indices = []
    for i in range(47):
        if i < mid:
            bit = 1 << i
            if not (best_mask_A & bit):
                indices.append(i)
        else:
            bit = 1 << (i - mid)
            if not (best_mask_B & bit):
                indices.append(i)
                
    if not indices:
        indices = [0]
        
    return sorted(indices)
