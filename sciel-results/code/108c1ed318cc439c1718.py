import sys
import time
from math import gcd
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s margin before 90s kill
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Compute L = lcm(2, ..., 48) using Python ints to avoid overflow
    L = 1
    for d in denominators:
        L = L * d // gcd(L, d)
        
    # Compute weights as Python ints
    weights = [L // d for d in denominators]
    
    # Split into two halves
    n1 = 23
    n2 = n - n1
    
    w1 = weights[:n1]
    w2 = weights[n1:]
    
    # Generate subset sums for first half
    # Using Python lists to handle big ints
    # We can optimize this by building iteratively
    sums1 = [0]
    for i in range(n1):
        w = w1[i]
        # Add w to all existing sums to get new sums, then extend
        sums1 = sums1 + [s + w for s in sums1]
        
    # Filter: keep only sums < L
    # Note: Since all weights are positive, we don't need to filter for > 0, but we do need < L
    # Actually, we just need to find max sum < L. 
    # It's more efficient to sort and use two pointers.
    
    # Check time
    if time.time() > deadline:
        return [0]
        
    # Generate subset sums for second half
    sums2 = [0]
    for i in range(n2):
        w = w2[i]
        sums2 = sums2 + [s + w for s in sums2]
        
    if time.time() > deadline:
        return [0]

    # Sort both lists
    sums1.sort()
    sums2.sort()
    
    # Two pointer scan for max sum < L
    # We want max(s1 + s2) such that s1 + s2 < L
    # Since lists are sorted ascending:
    # Start with smallest s1 and largest s2.
    # If s1 + s2 < L, update best, try larger s1 (to potentially get closer to L).
    # If s1 + s2 >= L, try smaller s2 (to reduce sum).
    
    best_sum = -1
    best_mask1 = 0
    best_mask2 = 0
    
    # We need to track masks as well. 
    # Re-doing generation with masks is better to avoid reconstructing from sum values which might collide.
    # Let's regenerate with masks.
    
    # Regenerate sums1 with masks
    data1 = [(0, 0)] # (sum, mask)
    for i in range(n1):
        w = w1[i]
        bit = 1 << i
        new_data = []
        for s, m in data1:
            new_data.append((s + w, m | bit))
        data1.extend(new_data)
        
    # Filter and sort data1
    # We only care about sums < L. 
    # Also, if multiple masks give same sum, we don't need both. But for two pointers, we just need the list of sums.
    # However, we need the mask to return indices.
    # Let's just store (sum, mask) pairs.
    
    if time.time() > deadline:
        return [0]

    data2 = [(0, 0)]
    for i in range(n2):
        w = w2[i]
        bit = 1 << i
        new_data = []
        for s, m in data2:
            new_data.append((s + w, m | bit))
        data2.extend(new_data)
        
    if time.time() > deadline:
        return [0]

    # Sort by sum
    data1.sort(key=lambda x: x[0])
    data2.sort(key=lambda x: x[0])
    
    # Two pointer on the sorted sums
    i = 0
    j = len(data2) - 1
    
    best_sum = -1
    best_m1 = 0
    best_m2 = 0
    
    while i < len(data1) and j >= 0:
        s1, m1 = data1[i]
        s2, m2 = data2[j]
        
        total = s1 + s2
        
        if total < L:
            if total > best_sum:
                best_sum = total
                best_m1 = m1
                best_m2 = m2
            i += 1
        else:
            j -= 1
            
    if time.time() > deadline:
        return [0]
         
    # Reconstruct indices
    indices = []
    for k in range(n1):
        if best_m1 & (1 << k):
            indices.append(k)
            
    for k in range(n2):
        if best_m2 & (1 << k):
            indices.append(n1 + k)
            
    if not indices:
        # Fallback: 1/2
        indices = [0]
        
    return indices

if __name__ == "__main__":
    # For local testing
    res = construct()
    denominators = list(range(2, 49))
    s = 0
    for idx in res:
        s += 1.0 / denominators[idx]
    print(f"Indices: {res}")
    print(f"Sum: {s}")
    print(f"Deficit: {1.0 - s}")
