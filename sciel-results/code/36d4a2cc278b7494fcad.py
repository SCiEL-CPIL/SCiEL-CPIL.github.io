import time
import bisect
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM of 1..48
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, range(1, 49))
    
    # Weights
    weights = [L // d for d in denominators]
    
    # Split into two halves
    mid = n // 2
    w1 = weights[:mid]
    w2 = weights[mid:]
    
    # Generate subset sums for w1
    # Returns list of (sum, mask)
    def gen_sums(weights):
        sums = [(0, 0)]
        for i, w in enumerate(weights):
            bit = 1 << i
            # Add new subsets
            # Using list comprehension for speed
            # We iterate over the current size to avoid adding to the list while iterating
            current_len = len(sums)
            for j in range(current_len):
                s, m = sums[j]
                sums.append((s + w, m | bit))
        return sums

    # Generate sums for both halves
    sums1 = gen_sums(w1)
    sums2 = gen_sums(w2)
    
    # Filter out sums >= L
    sums1 = [(s, m) for s, m in sums1 if s < L]
    sums2 = [(s, m) for s, m in sums2 if s < L]
    
    # Sort sums by sum value
    sums1.sort(key=lambda x: x[0])
    sums2.sort(key=lambda x: x[0])
    
    best_deficit = L # Max deficit is L (empty set)
    best_mask1 = 0
    best_mask2 = 0
    
    # Two-pointer search for max sum < L
    i = 0
    j = len(sums2) - 1
    
    while i < len(sums1) and j >= 0:
        s1, m1 = sums1[i]
        s2, m2 = sums2[j]
        current_sum = s1 + s2
        
        if current_sum < L:
            current_deficit = L - current_sum
            if current_deficit < best_deficit:
                best_deficit = current_deficit
                best_mask1 = m1
                best_mask2 = m2
                
                # Early exit if perfect
                if best_deficit == 1:
                    break
            
            # Try to increase the sum by moving to next larger s1
            i += 1
        else:
            # Sum is too large, reduce s2
            j -= 1
            
        # Time check
        if time.time() > DEADLINE:
            break

    # Reconstruct indices
    indices = []
    
    # Mask 1 corresponds to first 'mid' denominators
    for i in range(mid):
        if best_mask1 & (1 << i):
            indices.append(i)
            
    # Mask 2 corresponds to remaining denominators
    for i in range(len(w2)):
        if best_mask2 & (1 << i):
            indices.append(mid + i)
            
    # Ensure non-empty
    if not indices:
        # This shouldn't happen as single elements are valid and better than empty
        indices = [0]
        
    return indices
