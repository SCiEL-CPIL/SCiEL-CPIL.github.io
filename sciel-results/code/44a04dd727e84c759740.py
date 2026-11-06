import sys
from math import gcd
from bisect import bisect_right

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
    # sums1[i] is sum of subset represented by mask i
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
            
    # Filter and prepare second half for binary search
    # We need (sum, mask) pairs where sum < L
    # To save memory, we can store sums in a list and masks in another,
    # or just filter and sort.
    
    # Let's create a list of valid (sum, mask) for the second half
    # But storing tuples is memory heavy.
    # Let's store sums in a list `valid_sums2` and masks in `valid_masks2`.
    
    valid_sums2 = []
    valid_masks2 = []
    
    for mask in range(1 << n2):
        s = sums2[mask]
        if s < L:
            valid_sums2.append(s)
            valid_masks2.append(mask)
            
    # Sort by sum
    # We need to sort pairs. 
    # We can sort indices or use zip.
    # zip creates tuples. 
    # Let's sort a list of indices based on sums2?
    # Or just sort the two lists together?
    # Python doesn't have sort two lists together easily without zip.
    # Let's use a list of tuples for simplicity, memory should be okay if filtered.
    
    # Re-evaluating memory: 
    # If valid_sums2 has ~1/4 of 16M = 4M items.
    # 4M tuples is ~300MB. Okay.
    
    pairs2 = list(zip(valid_sums2, valid_masks2))
    pairs2.sort(key=lambda x: x[0])
    
    # Extract sorted sums for bisect
    sorted_sums2 = [p[0] for p in pairs2]
    
    best_sum = -1
    best_mask1 = 0
    best_mask2 = 0
    
    # Iterate over first half
    for mask1 in range(1 << n1):
        s1 = sums1[mask1]
        if s1 >= L:
            continue
            
        target = L - 1 - s1
        
        # Find largest sum in second half <= target
        # bisect_right gives index of first element > target
        # So index - 1 is the largest <= target
        idx = bisect_right(sorted_sums2, target)
        
        if idx > 0:
            # The best candidate in second half is at idx - 1
            s2, m2 = pairs2[idx - 1]
            total_sum = s1 + s2
            
            if total_sum > best_sum:
                best_sum = total_sum
                best_mask1 = mask1
                best_mask2 = m2
                
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
        # Fallback: pick one with smallest reciprocal? No, must be < 1.
        # 1/2 is < 1.
        # But our search should find something.
        # If L=2 (not here), 1/2 is 0.5.
        # Here L is huge.
        # If no subset found, return [0] (1/2)
        indices = [0]
        
    return indices
