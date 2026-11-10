import time
import numpy as np
from math import gcd

def construct():
    start_time = time.time()
    deadline = 85.0
    
    # 1. Calculate LCM of 2..48
    L = 1
    for i in range(2, 49):
        L = L * i // gcd(L, i)
        
    # 2. Weights
    # Indices 0..46 correspond to denominators 2..48
    weights = np.array([L // (i + 2) for i in range(47)], dtype=np.int64)
    
    # Split
    mid = 23
    weights_A = weights[:mid]
    weights_B = weights[mid:]
    
    def gen_sums_masks(weights_list):
        n = len(weights_list)
        # Start with sum 0, mask 0
        sums = np.array([0], dtype=np.int64)
        masks = np.array([0], dtype=np.int64)
        
        for k in range(n):
            w = weights_list[k]
            # New sums are existing sums + w
            # New masks are existing masks with bit k set
            new_sums = sums + w
            new_masks = masks | (1 << k)
            
            # Concatenate
            sums = np.concatenate([sums, new_sums])
            masks = np.concatenate([masks, new_masks])
            
            # Filter out sums >= L
            # This keeps the array size manageable if many sums exceed L
            # However, for small k, few sums exceed L.
            # For large k, many might.
            # But 2^24 is the max size.
            # We can filter at the end to avoid repeated filtering overhead?
            # No, filtering early reduces the size of subsequent concatenations?
            # Wait, if we filter early, we might miss combinations?
            # No, if sum >= L, adding more positive weights will keep it >= L.
            # So we can safely discard sums >= L at each step.
            valid = sums < L
            sums = sums[valid]
            masks = masks[valid]
            
            if time.time() - start_time > deadline:
                # Fallback: return a simple greedy solution
                return None, None
                
        return sums, masks

    sums_A, masks_A = gen_sums_masks(weights_A)
    if sums_A is None:
        # Fallback
        return [0] # Just index 0 (1/2) is valid and non-empty
        
    sums_B, masks_B = gen_sums_masks(weights_B)
    if sums_B is None:
        return [0]

    # 3. Sort
    # Sort A
    order_A = np.argsort(sums_A)
    sums_A_sorted = sums_A[order_A]
    masks_A_sorted = masks_A[order_A]
    
    # Sort B
    order_B = np.argsort(sums_B)
    sums_B_sorted = sums_B[order_B]
    masks_B_sorted = masks_B[order_B]
    
    # 4. Two-pointer search
    best_sum = -1
    best_mask_A = 0
    best_mask_B = 0
    
    i = 0
    j = len(sums_B_sorted) - 1
    
    n_A = len(sums_A_sorted)
    n_B = len(sums_B_sorted)
    
    while i < n_A and j >= 0:
        current_sum = sums_A_sorted[i] + sums_B_sorted[j]
        if current_sum < L:
            if current_sum > best_sum:
                best_sum = current_sum
                best_mask_A = masks_A_sorted[i]
                best_mask_B = masks_B_sorted[j]
            i += 1
        else:
            j -= 1
            
    # 5. Construct indices
    indices = []
    for k in range(len(weights_A)):
        if best_mask_A & (1 << k):
            indices.append(k)
            
    for k in range(len(weights_B)):
        if best_mask_B & (1 << k):
            indices.append(mid + k)
            
    return sorted(indices)
