import time
import numpy as np
from math import lcm

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)

    # Calculate LCM of all denominators
    L = 1
    for d in denominators:
        L = lcm(L, d)

    # Weights are L // d
    weights = [L // d for d in denominators]

    def check_time():
        return time.time() > deadline

    # Split into two halves
    mid = n // 2
    half1_idx = list(range(mid))
    half2_idx = list(range(mid, n))

    w1 = [weights[i] for i in half1_idx]
    w2 = [weights[i] for i in half2_idx]

    def gen_sums_numpy(weights_list):
        """Generate all subset sums < L with their masks using numpy"""
        # Start with empty subset
        sums = np.array([0], dtype=np.int64)
        masks = np.array([0], dtype=np.int64)
        
        for i, w in enumerate(weights_list):
            if check_time():
                break
            bit = 1 << i
            # Create new sums by adding w to existing sums
            # We need to be careful with memory
            new_sums = np.empty(len(sums) * 2, dtype=np.int64)
            new_masks = np.empty(len(sums) * 2, dtype=np.int64)
            
            new_sums[:len(sums)] = sums
            new_sums[len(sums):] = sums + w
            
            new_masks[:len(sums)] = masks
            new_masks[len(sums):] = masks | bit
            
            sums = new_sums
            masks = new_masks
            
            # Filter out sums >= L
            # This reduces size, but we can also filter at the end
            # Filtering at each step keeps memory lower
            valid_mask = sums < L
            if not np.all(valid_mask):
                sums = sums[valid_mask]
                masks = masks[valid_mask]
                
        return sums, masks

    # 1. Generate sums with masks
    sums1, masks1 = gen_sums_numpy(w1)
    if check_time():
        return [0]
    
    sums2, masks2 = gen_sums_numpy(w2)
    if check_time():
        return [0]

    # 2. Sort sums and masks together
    # np.argsort is fast
    idx1 = np.argsort(sums1)
    sums1_sorted = sums1[idx1]
    masks1_sorted = masks1[idx1]
    
    idx2 = np.argsort(sums2)
    sums2_sorted = sums2[idx2]
    masks2_sorted = masks2[idx2]

    # 3. Find Max Sum < L
    # Iterate through sums2, find best sums1
    best_sum = -1
    best_mask1 = 0
    best_mask2 = 0
    
    # We can vectorize the searchsorted?
    # target = L - 1 - sums2_sorted
    # idxs = np.searchsorted(sums1_sorted, target, side='right') - 1
    # valid = idxs >= 0
    # candidates = sums1_sorted[idxs[valid]] + sums2_sorted[valid]
    # best_idx = np.argmax(candidates)
    # This is much faster!
    
    target = L - 1 - sums2_sorted
    idxs = np.searchsorted(sums1_sorted, target, side='right') - 1
    
    valid = idxs >= 0
    if not np.any(valid):
        return [0]
        
    # Get valid indices
    valid_idxs = np.where(valid)[0]
    
    # Calculate candidate sums
    # sums1_sorted[idxs[valid_idxs]] + sums2_sorted[valid_idxs]
    s1_candidates = sums1_sorted[idxs[valid_idxs]]
    s2_candidates = sums2_sorted[valid_idxs]
    
    total_candidates = s1_candidates + s2_candidates
    
    # Find the maximum
    best_pos = np.argmax(total_candidates)
    best_sum = total_candidates[best_pos]
    
    # Get the masks
    # The index in the original sorted arrays
    pos_in_sorted2 = valid_idxs[best_pos]
    pos_in_sorted1 = idxs[pos_in_sorted2]
    
    best_mask2 = masks2_sorted[pos_in_sorted2]
    best_mask1 = masks1_sorted[pos_in_sorted1]

    if best_sum == -1:
        # Fallback
        return [0]

    # 4. Reconstruct indices
    final_indices = []
    for k in range(mid):
        if best_mask1 & (1 << k):
            final_indices.append(k)
    for k in range(mid, n):
        if best_mask2 & (1 << (k - mid)):
            final_indices.append(k - mid)
            
    if not final_indices:
        return [0]

    return final_indices
