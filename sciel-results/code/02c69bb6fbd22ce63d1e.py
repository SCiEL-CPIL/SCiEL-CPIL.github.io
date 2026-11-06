import time
from math import lcm
from array import array

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM
    L = 1
    for d in denominators:
        L = lcm(L, d)
        
    # Calculate weights
    weights = [L // d for d in denominators]
    
    # Split into two halves
    mid = 23
    n1 = mid
    n2 = n - mid
    
    # Generate subset sums for first half (0 to 22)
    # sums1: list of int
    # masks1: array of unsigned int (32-bit is enough for 23 bits)
    sums1 = [0]
    masks1 = array('I', [0])
    
    for i in range(n1):
        w = weights[i]
        bit = 1 << i
        current_len = len(sums1)
        new_sums = [0] * current_len
        new_masks = array('I', [0]) * current_len
        
        for idx in range(current_len):
            ns = sums1[idx] + w
            if ns < L:
                new_sums[idx] = ns
                new_masks[idx] = masks1[idx] | bit
            else:
                # If sum >= L, we can't use this branch for valid subsets
                # We effectively skip it by not extending, but we pre-allocated.
                # Better to build lists dynamically or filter.
                pass
                
        # This pre-allocation is messy if we filter.
        # Let's use dynamic append for clarity and safety, accepting slight speed loss.
        
    # Re-do generation with dynamic append
    sums1 = [0]
    masks1 = array('I', [0])
    
    for i in range(n1):
        w = weights[i]
        bit = 1 << i
        current_len = len(sums1)
        # Iterate over existing sums
        for idx in range(current_len):
            ns = sums1[idx] + w
            if ns < L:
                sums1.append(ns)
                masks1.append(masks1[idx] | bit)
                
        if time.time() > deadline - 30:
            break
            
    # Generate subset sums for second half (23 to 46)
    sums2 = [0]
    masks2 = array('I', [0])
    
    for i in range(n2):
        w = weights[mid + i]
        bit = 1 << i
        current_len = len(sums2)
        for idx in range(current_len):
            ns = sums2[idx] + w
            if ns < L:
                sums2.append(ns)
                masks2.append(masks2[idx] | bit)
                
        if time.time() > deadline - 30:
            break
            
    # Sort second half by sum
    # We need to sort sums2 and keep masks2 aligned.
    # Create list of indices sorted by sum
    # To save memory, we can create a list of (sum, mask) tuples? 
    # Or sort indices.
    
    # Sorting 16M indices is memory heavy.
    # Let's try sorting a list of tuples (sum, mask) for half 2.
    # If memory is an issue, we can split half 2 further?
    # For now, assume memory is okay.
    
    # Optimization: Convert to list of tuples, sort, then extract sums for bisect.
    # items2 = list(zip(sums2, masks2))
    # items2.sort(key=lambda x: x[0])
    # sorted_sums2 = [x[0] for x in items2]
    # sorted_masks2 = [x[1] for x in items2]
    
    # This creates 3 large lists.
    # Alternative:
    # idx2 = sorted(range(len(sums2)), key=lambda i: sums2[i])
    # This creates one large list of ints (indices).
    
    idx2 = sorted(range(len(sums2)), key=lambda i: sums2[i])
    sorted_sums2 = [sums2[i] for i in idx2]
    # We don't need to extract masks into a new list immediately, 
    # we can access masks2[idx2[pos]] when we find the best match.
    
    best_sum = -1
    best_m1 = 0
    best_m2 = 0
    
    # Binary search for each sum in half 1
    import bisect
    
    for idx1 in range(len(sums1)):
        if time.time() > deadline - 10:
            break
            
        s1 = sums1[idx1]
        m1 = masks1[idx1]
        target = L - 1 - s1
        
        # Find largest s2 <= target
        pos = bisect.bisect_right(sorted_sums2, target)
        if pos > 0:
            s2 = sorted_sums2[pos - 1]
            # Get corresponding mask
            # idx2[pos-1] is the index in original sums2/masks2
            orig_idx = idx2[pos - 1]
            m2 = masks2[orig_idx]
            
            total = s1 + s2
            if total > best_sum:
                best_sum = total
                best_m1 = m1
                best_m2 = m2
                
    # Reconstruct indices
    result_indices = []
    for i in range(n1):
        if best_m1 & (1 << i):
            result_indices.append(i)
            
    for i in range(n2):
        if best_m2 & (1 << i):
            result_indices.append(mid + i)
            
    if not result_indices:
        return [0]
        
    return result_indices
