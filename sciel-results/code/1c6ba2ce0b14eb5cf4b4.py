def construct():
    from fractions import Fraction
    import time
    import math

    deadline = time.time() + 80  # 80 second deadline for safety (90s limit)

    # Denominators are 2 to 48, indices 0 to 46
    denoms = list(range(2, 49))
    n = len(denoms)
    fracs = [Fraction(1, d) for d in denoms]

    # Meet-in-the-middle approach
    # Split into two halves
    mid = n // 2  # 23
    left_denoms = denoms[:mid]
    right_denoms = denoms[mid:]
    left_fracs = fracs[:mid]
    right_fracs = fracs[mid:]
    left_n = mid
    right_n = n - mid

    # Generate all subset sums for left half
    # For each subset, store (sum, bitmask) where bitmask indicates which indices are selected
    left_sums = {}  # sum -> list of bitmasks (we only need the best one per sum, but let's collect)
    
    # Actually, for MITM we need to find for each right subset sum s_r,
    # the largest left subset sum s_l such that s_l + s_r < 1
    # So we want s_l < 1 - s_r
    
    # Generate all 2^23 left subsets
    left_entries = []  # list of (sum, bitmask)
    for mask in range(1 << left_n):
        s = Fraction(0)
        # Compute sum for this mask
        # Optimization: use Gray code or incremental, but 2^23 = 8.4M, let's be careful
        # Let's compute incrementally
        # Actually, let's just iterate
        m = mask
        idx = 0
        while m:
            if m & 1:
                s += left_fracs[idx]
                if s >= 1:
                    break
            m >>= 1
            idx += 1
        if s < 1:
            left_entries.append((s, mask))
    
    if time.time() > deadline - 5:
        # Fallback: greedy
        chosen = []
        total = Fraction(0)
        for i in range(n):
            if total + fracs[i] < 1:
                chosen.append(i)
                total += fracs[i]
        return chosen if chosen else [0]

    # Sort left_entries by sum descending
    left_entries.sort(key=lambda x: x[0], reverse=True)
    
    # For each right subset, find the best left subset
    # We need s_l < 1 - s_r, maximize s_l (to minimize deficit)
    # Since left_entries is sorted descending, we can binary search
    
    # Actually, let's build a list of just the sums for binary search
    left_sums_sorted = [s for s, mask in left_entries]
    left_masks_sorted = [mask for s, mask in left_entries]
    
    # left_sums_sorted is in descending order
    # We want the largest s_l such that s_l < 1 - s_r
    # This is equivalent to finding the first element in the sorted list that is < 1 - s_r
    # But since it's descending, we want the rightmost element that is < 1 - s_r
    # Or rather, we want the maximum s_l < 1 - s_r
    
    # Let's use bisect. But bisect works on ascending order.
    # Let's reverse to ascending.
    left_sums_asc = left_sums_sorted[::-1]
    left_masks_asc = left_masks_sorted[::-1]
    
    import bisect
    
    best_deficit = Fraction(1, 1)
    best_left_mask = 0
    best_right_mask = 0
    best_total = Fraction(0)
    
    # Generate all right subsets
    for rmask in range(1 << right_n):
        s_r = Fraction(0)
        m = rmask
        idx = 0
        while m:
            if m & 1:
                s_r += right_fracs[idx]
                if s_r >= 1:
                    break
            m >>= 1
            idx += 1
        
        if s_r >= 1:
            continue
        
        # We need s_l < 1 - s_r
        threshold = Fraction(1, 1) - s_r
        
        # Find the largest s_l in left_sums_asc that is < threshold
        # bisect_left gives the first index where left_sums_asc[i] >= threshold
        # So the element before that is the largest one < threshold
        # But we want strictly less than 1, so s_l + s_r < 1 means s_l < 1 - s_r
        # If 1 - s_r is exactly in the list, we need to exclude it
        
        pos = bisect.bisect_left(left_sums_asc, threshold)
        # left_sums_asc[pos] >= threshold
        # So left_sums_asc[pos-1] < threshold (if pos > 0)
        # We want the maximum s_l < threshold
        # That is left_sums_asc[pos-1] if pos > 0
        
        if pos > 0:
            s_l = left_sums_asc[pos - 1]
            lmask = left_masks_asc[pos - 1]
            
            total = s_l + s_r
            deficit = Fraction(1, 1) - total
            
            if deficit > 0 and deficit < best_deficit:
                best_deficit = deficit
                best_left_mask = lmask
                best_right_mask = rmask
                best_total = total
        
        # Also check if the subset with just right (no left) is better
        # That's s_l = 0, which is in the list (mask=0)
        # pos for threshold > 0 will always be >= 1 since 0 is in the list and 0 < threshold (since s_r < 1)
        # So s_l = 0 is covered when pos-1 points to it
        
        # Check time
        if rmask % 1000000 == 0:
            if time.time() > deadline - 5:
                break

    # Reconstruct the chosen indices
    chosen = []
    
    # Add left indices
    lmask = best_left_mask
    for i in range(left_n):
        if lmask & (1 << i):
            chosen.append(i)
    
    # Add right indices
    rmask = best_right_mask
    for i in range(right_n):
        if rmask & (1 << i):
            chosen.append(mid + i)
    
    # If no subset found (shouldn't happen since empty set is valid but we need non-empty)
    if not chosen:
        # Try to find any valid non-empty subset
        # The empty set has sum 0, deficit 1
        # We need at least one element
        # Find the single element with largest fraction (smallest denominator) that is < 1
        # All 1/d for d >= 2 are < 1
        # So just pick index 0 (denominator 2)
        chosen = [0]
    
    # Verify
    total = Fraction(0)
    for i in chosen:
        total += fracs[i]
    
    if total >= 1:
        # Fallback
        chosen = [0]
    
    return chosen
