import time
from math import gcd
from bisect import bisect_left

def construct():
    deadline = time.time() + 55  # Safety margin
    
    denoms = list(range(2, 49))
    N = len(denoms)
    
    # Calculate LCM of all denominators
    L = 1
    for d in denoms:
        L = L * d // gcd(L, d)
        
    # Weights
    weights = [L // d for d in denoms]
    
    # Split into two halves
    mid = N // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    # Generate subset sums for first half
    # List of (sum, mask)
    sums1 = [(0, 0)]
    for i in range(len(w1)):
        wi = w1[i]
        # Create new sums by adding wi to existing sums
        # Optimization: only keep sums < L
        new_sums = []
        for s, m in sums1:
            ns = s + wi
            if ns < L:
                new_sums.append((ns, m | (1 << i)))
        sums1.extend(new_sums)
        
    # Sort sums1 by sum value
    sums1.sort(key=lambda x: x[0])
    
    # Extract sorted sums and masks for binary search
    # We need to binary search on the sums.
    # bisect works on lists. If we have a list of tuples, it sorts by first element then second.
    # We want to find the largest sum < limit.
    # We can create a separate list of just the sums for bisect?
    # Or use bisect on the list of tuples with a dummy mask?
    # Better: create a list of sums only, and a parallel list of masks?
    # Or just keep the list of tuples and use bisect_left with (limit, -1)?
    # If sums1 is sorted by sum, bisect_left(sums1, (limit, -1)) gives the position of the first element >= limit.
    # The element before that is the largest < limit.
    
    best_sum = 0
    best_mask1 = 0
    best_mask2 = 0
    
    # Iterate through subsets of second half
    # We can generate them on the fly to save memory?
    # Or generate a list? 2^24 is 16M. A list of 16M integers is ~128MB. Might be tight but okay.
    # Generating on the fly is safer for memory.
    
    # However, we need to be fast.
    # Let's generate sums2 recursively or iteratively.
    # Iterative generation of 2^24 items in a loop is fine.
    
    # We can also prune: if current sum >= L, skip.
    
    # To speed up, we can generate sums2 in a list too, filter, and then iterate?
    # No, iterating 16M times and doing a binary search (C speed) is fast enough.
    
    for mask2 in range(1 << len(w2)):
        if time.time() > deadline:
            break
            
        # Calculate sum for this mask
        # Optimization: use a recursive generator or iterative DP?
        # Simple loop for 24 bits is 24 operations. 16M * 24 = 400M ops. Might be too slow in Python.
        # Better to generate sums2 list first using the same extend method as sums1.
        
    # Let's generate sums2 list first.
    sums2 = [(0, 0)]
    for i in range(len(w2)):
        wi = w2[i]
        new_sums = []
        for s, m in sums2:
            ns = s + wi
            if ns < L:
                new_sums.append((ns, m | (1 << i)))
        sums2.extend(new_sums)
        
    # Now iterate through sums2
    # sums2 is not sorted, but we don't need it to be.
    # We just need to find best s1 for each s2.
    
    for s2, m2 in sums2:
        if time.time() > deadline:
            break
            
        # We need s1 < L - s2
        limit = L - s2
        
        # Find largest s1 < limit in sums1
        # sums1 is sorted by sum.
        # bisect_left returns index where limit would be inserted.
        # If limit is 100, and sums are [10, 20, 30], bisect_left([10,20,30], 100) returns 3.
        # We want the element at index 2 (30).
        # If limit is 25, bisect_left returns 2. Element at 1 is 20.
        # So we look at idx = bisect_left(sums1, (limit, -1))
        # The candidate is at idx - 1.
        
        idx = bisect_left(sums1, (limit, -1))
        
        if idx > 0:
            s1, m1 = sums1[idx - 1]
            total_sum = s1 + s2
            if total_sum > best_sum:
                best_sum = total_sum
                best_mask1 = m1
                best_mask2 = m2
                
    # Construct result indices
    result_idxs = []
    
    # Map mask1 back to original indices 0..mid-1
    for i in range(len(w1)):
        if best_mask1 & (1 << i):
            result_idxs.append(i)
            
    # Map mask2 back to original indices mid..N-1
    for i in range(len(w2)):
        if best_mask2 & (1 << i):
            result_idxs.append(mid + i)
            
    # Ensure non-empty
    if not result_idxs:
        return [0]
        
    return result_idxs
