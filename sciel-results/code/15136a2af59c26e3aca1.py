import math
import numpy as np
from fractions import Fraction

def construct():
    denominators = list(range(2, 49))
    N = len(denominators)
    
    # Calculate LCM
    L = 1
    for d in denominators:
        L = L * d // math.gcd(L, d)
        
    # Weights
    W = [L // d for d in denominators]
    
    # Target sum strictly less than L
    Target = L - 1
    
    # Meet in the middle
    mid = N // 2
    W1 = W[:mid]
    W2 = W[mid:]
    
    # Generate subset sums for W1
    sums1 = [0]
    masks1 = [0]
    for i, w in enumerate(W1):
        # Use list comprehension or loop
        # To avoid memory spikes, we can extend
        current_len = len(sums1)
        for j in range(current_len):
            sums1.append(sums1[j] + w)
            masks1.append(masks1[j] | (1 << i))
            
    # Sort sums1 and keep masks aligned
    # Create list of tuples, sort, then unzip? Or sort indices?
    # Sorting 8 million elements in Python might take a few seconds.
    # Let's try to be efficient.
    pairs1 = list(zip(sums1, masks1))
    pairs1.sort(key=lambda x: x[0])
    sums1 = [p[0] for p in pairs1]
    masks1 = [p[1] for p in pairs1]
    
    best_sum = 0
    best_mask1 = 0
    best_mask2 = 0
    
    # Generate subset sums for W2 and check
    sums2 = [0]
    masks2 = [0]
    for i, w in enumerate(W2):
        current_len = len(sums2)
        for j in range(current_len):
            sums2.append(sums2[j] + w)
            masks2.append(masks2[j] | (1 << i))
            
    # Binary search for best combination
    # We iterate through all subsets of W2
    # For each sum2, we want max sum1 such that sum1 + sum2 <= Target
    # sums1 is sorted.
    
    import bisect
    
    for i in range(len(sums2)):
        s2 = sums2[i]
        if s2 > Target:
            continue
        rem = Target - s2
        # Find rightmost index in sums1 with value <= rem
        # bisect_right returns insertion point to the right of existing entries
        # We want the largest value <= rem.
        # If we use bisect_right, it gives index after the last element <= rem.
        # So index - 1 is the candidate.
        idx = bisect.bisect_right(sums1, rem) - 1
        if idx >= 0:
            s1 = sums1[idx]
            total = s1 + s2
            if total > best_sum:
                best_sum = total
                best_mask1 = masks1[idx]
                best_mask2 = masks2[i]
                
    # Reconstruct indices
    # mask1 corresponds to W1 (indices 0 to mid-1)
    # mask2 corresponds to W2 (indices mid to N-1)
    
    selected = []
    for i in range(mid):
        if best_mask1 & (1 << i):
            selected.append(i)
            
    for i in range(N - mid):
        if best_mask2 & (1 << i):
            selected.append(mid + i)
            
    return selected
