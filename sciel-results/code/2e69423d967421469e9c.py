import sys
from math import gcd
import numpy as np

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Compute L = lcm(2, ..., 48)
    L = 1
    for d in denominators:
        L = L * d // gcd(L, d)
        
    # Compute weights
    weights = [L // d for d in denominators]
    
    # Split into two halves
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    n1 = len(w1)
    n2 = len(w2)
    
    # Generate subset sums for first half
    sums1 = [0] * (1 << n1)
    for i in range(n1):
        w = w1[i]
        for j in range(1 << i):
            sums1[(1 << i) | j] = sums1[j] + w
            
    # Generate subset sums for second half
    sums2 = [0] * (1 << n2)
    for i in range(n2):
        w = w2[i]
        for j in range(1 << i):
            sums2[(1 << i) | j] = sums2[j] + w
            
    # Prepare second half for binary search using NumPy
    # Filter sums < L
    # We need to keep track of masks
    # Create arrays for valid entries
    valid_sums2_list = []
    valid_masks2_list = []
    
    # To avoid slow Python loop for filtering 16M items, we can use NumPy
    sums2_arr = np.array(sums2, dtype=np.int64)
    masks2_arr = np.arange(1 << n2, dtype=np.int64)
    
    mask_valid2 = sums2_arr < L
    valid_sums2 = sums2_arr[mask_valid2]
    valid_masks2 = masks2_arr[mask_valid2]
    
    # Sort valid_sums2 and valid_masks2 together
    # np.argsort returns indices that would sort the array
    sort_idx = np.argsort(valid_sums2)
    sorted_sums2 = valid_sums2[sort_idx]
    sorted_masks2 = valid_masks2[sort_idx]
    
    # Prepare first half
    sums1_arr = np.array(sums1, dtype=np.int64)
    masks1_arr = np.arange(1 << n1, dtype=np.int64)
    
    mask_valid1 = sums1_arr < L
    valid_sums1 = sums1_arr[mask_valid1]
    valid_masks1 = masks1_arr[mask_valid1]
    
    if len(valid_sums1) == 0 or len(sorted_sums2) == 0:
        return [0] # Fallback, should not happen
    
    # For each sum in valid_sums1, find the largest sum in sorted_sums2 
    # such that sum1 + sum2 < L  => sum2 < L - sum1
    # Target max sum2 is L - 1 - sum1
    targets = L - 1 - valid_sums1
    
    # np.searchsorted(sorted_sums2, targets, side='right') - 1
    # side='right' returns index of first element > target
    # So idx - 1 is the largest element <= target
    idxs = np.searchsorted(sorted_sums2, targets, side='right') - 1
    
    # Filter out invalid indices (idx < 0)
    valid_idxs = idxs[idxs >= 0]
    if len(valid_idxs) == 0:
        return [0]
        
    # Get corresponding sums and masks
    # Note: valid_sums1 needs to be aligned with valid_idxs
    # valid_idxs are indices into the original valid_sums1 array?
    # No, idxs are indices into sorted_sums2.
    # The mask1 corresponding to valid_idxs[i] is valid_masks1[i].
    # But we filtered idxs to only keep those >= 0.
    # We need to keep the corresponding valid_sums1 and valid_masks1.
    
    # Let's re-align
    # valid_idxs is a subset of indices from 0 to len(valid_sums1)-1?
    # No, idxs is an array of same length as valid_sums1.
    # idxs[i] is the index in sorted_sums2 for valid_sums1[i].
    # So we need to select elements from valid_sums1 and valid_masks1 at positions where idxs >= 0.
    
    mask_ok = idxs >= 0
    valid_sums1_ok = valid_sums1[mask_ok]
    valid_masks1_ok = valid_masks1[mask_ok]
    valid_idxs_ok = idxs[mask_ok]
    
    # Now compute total sums
    total_sums = valid_sums1_ok + sorted_sums2[valid_idxs_ok]
    
    # Find max total sum
    max_idx = np.argmax(total_sums)
    best_sum1 = valid_sums1_ok[max_idx]
    best_mask1 = valid_masks1_ok[max_idx]
    best_mask2 = sorted_masks2[valid_idxs_ok[max_idx]]
    
    # Reconstruct indices
    indices = []
    for i in range(n1):
        if best_mask1 & (1 << i):
            indices.append(i)
            
    for i in range(n2):
        if best_mask2 & (1 << i):
            indices.append(mid + i)
            
    # Ensure non-empty
    if not indices:
        indices = [0]
        
    return indices
