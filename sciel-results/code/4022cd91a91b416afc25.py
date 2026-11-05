import time
from math import gcd
from functools import reduce
import bisect
from fractions import Fraction

def construct():
    """
    Finds the subset of denominators [2, 3, ..., 48] that minimizes the deficit 
    1 - sum(1/d) subject to sum(1/d) < 1.
    
    Uses a meet-in-the-middle approach with exact integer arithmetic.
    """
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s kill

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of all denominators to work with integers
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denominators)
    
    # Values in integer scale: V[i] = L // denominators[i]
    V = [L // d for d in denominators]
    
    # Split into two halves for meet-in-the-middle
    # n=47. Let's split 23 and 24.
    n1 = 23
    n2 = n - n1
    V1 = V[:n1]
    V2 = V[n1:]
    
    # Generate all subset sums for V1
    # There are 2^23 = 8,388,608 subsets.
    # We use a list of tuples (sum, mask) to keep track of which elements were chosen.
    # However, storing masks for 8M items is memory heavy but manageable.
    # Alternatively, we can just store sums and then reconstruct masks if needed? 
    # No, we need the mask to return indices.
    
    # To save memory and time, let's generate sums for V2 first, sort them, 
    # and keep track of the best mask for each sum? No, multiple masks can have the same sum.
    # But we only care about the maximum sum < L.
    # Actually, for a fixed sum value, any mask is equivalent for the total sum.
    # So we can just store the sums of V2 in a sorted list, and for each sum, 
    # we might need a representative mask.
    # But wait, if we have multiple masks with the same sum, they are interchangeable 
    # regarding the total sum. So we can just keep one mask per unique sum?
    # Or even better, just keep the list of (sum, mask) pairs, sort by sum, 
    # and when we find the best total, we pick the corresponding mask.
    
    # Let's generate sums for V2 (size 24, 16M items). This might be too slow/memory intensive in Python.
    # 16M big integers is a lot.
    # Let's try to be smarter.
    
    # Alternative: Generate sums for V1 (8M) and V2 (16M).
    # 16M big integers in Python is ~16M * 100 bytes = 1.6GB. Might be tight.
    # Let's try to optimize the generation.
    
    def gen_sums(vals):
        sums = [0]
        masks = [0]
        for i, v in enumerate(vals):
            if time.time() > deadline:
                return sums, masks
            # Optimization: list comprehension is faster
            new_sums = [s + v for s in sums]
            new_masks = [m | (1 << i) for m in masks]
            sums.extend(new_sums)
            masks.extend(new_masks)
        return sums, masks

    # Generate V2 sums first as they are larger? Or V1?
    # Let's generate V1 (23 items) and V2 (24 items).
    
    sums1, masks1 = gen_sums(V1)
    
    if time.time() > deadline:
        return [0] 

    sums2, masks2 = gen_sums(V2)
    
    if time.time() > deadline:
        return [0]

    # Sort sums2 and masks2 together by sum
    pairs2 = list(zip(sums2, masks2))
    pairs2.sort(key=lambda x: x[0])
    
    # Extract sorted sums for bisect
    sorted_sums2 = [p[0] for p in pairs2]
    
    best_total = -1
    best_mask1 = 0
    best_mask2 = 0
    
    # Iterate through sums1
    for i in range(len(sums1)):
        s1 = sums1[i]
        if s1 >= L:
            continue
            
        # We want s2 such that s1 + s2 < L
        # s2 < L - s1
        # s2 <= L - 1 - s1
        target = L - 1 - s1
        
        # Find rightmost index in sorted_sums2 <= target
        idx = bisect.bisect_right(sorted_sums2, target)
        
        if idx > 0:
            s2 = sorted_sums2[idx - 1]
            m2 = pairs2[idx - 1][1]
            
            total = s1 + s2
            if total > best_total:
                best_total = total
                best_mask1 = masks1[i]
                best_mask2 = m2
                
        if time.time() > deadline:
            break

    # Reconstruct indices
    indices = []
    for j in range(n1):
        if best_mask1 & (1 << j):
            indices.append(j)
            
    for j in range(n2):
        if best_mask2 & (1 << j):
            indices.append(n1 + j)
            
    if not indices:
        return [0]

    # Verify result
    current_sum_frac = Fraction(0, 1)
    for idx in indices:
        current_sum_frac += Fraction(1, denominators[idx])
        
    if current_sum_frac >= 1:
        # Should not happen, but fallback
        return [0]

    return indices

if __name__ == "__main__":
    res = construct()
    pass
