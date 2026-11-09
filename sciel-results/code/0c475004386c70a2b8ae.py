import math
import bisect
from math import gcd

def construct():
    denominators = list(range(2, 49))
    N = len(denominators)
    
    # Calculate LCM
    L = 1
    for d in denominators:
        L = L * d // math.gcd(L, d)
        
    # Weights
    W = [L // d for d in denominators]
    
    # Split
    mid = N // 2
    W1 = W[:mid]
    W2 = W[mid:]
    
    N1 = len(W1)
    N2 = len(W2)
    
    # Generate subset sums for W1
    # We don't need to sort W1, we just iterate over it
    sums1 = [0]
    masks1 = [0]
    for i, w in enumerate(W1):
        current_len = len(sums1)
        for j in range(current_len):
            sums1.append(sums1[j] + w)
            masks1.append(masks1[j] | (1 << i))
            
    # Generate subset sums for W2
    sums2 = [0]
    masks2 = [0]
    for i, w in enumerate(W2):
        current_len = len(sums2)
        for j in range(current_len):
            sums2.append(sums2[j] + w)
            masks2.append(masks2[j] | (1 << i))
            
    # Create pairs for W2 and sort
    pairs2 = list(zip(sums2, masks2))
    del sums2, masks2
    pairs2.sort(key=lambda x: x[0])
    
    # Extract sorted sums for binary search
    sorted_sums2 = [p[0] for p in pairs2]
    # Keep pairs2 to access masks
    
    best_sum = -1
    best_mask1 = 0
    best_mask2 = 0
    
    # Iterate over W1 subsets
    for i in range(len(sums1)):
        s1 = sums1[i]
        if s1 >= L:
            continue
            
        rem = L - 1 - s1 # We need s1 + s2 <= L - 1
        
        # Find rightmost index in sorted_sums2 with value <= rem
        # bisect_right returns insertion point to the right of existing entries
        # So index - 1 is the candidate
        idx = bisect.bisect_right(sorted_sums2, rem) - 1
        
        if idx >= 0:
            s2 = sorted_sums2[idx]
            total = s1 + s2
            if total > best_sum:
                best_sum = total
                best_mask1 = masks1[i]
                best_mask2 = pairs2[idx][1]
                
    # Reconstruct indices
    selected = []
    for i in range(N1):
        if best_mask1 & (1 << i):
            selected.append(i)
            
    for i in range(N2):
        if best_mask2 & (1 << i):
            selected.append(mid + i)
            
    return selected
