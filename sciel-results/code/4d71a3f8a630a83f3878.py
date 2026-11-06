import time
import math
import numpy as np

def construct():
    start_time = time.time()
    timeout = 80.0
    
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Compute LCM
    L = 1
    for d in denominators:
        L = L * d // math.gcd(L, d)
        
    # Weights
    weights = [L // d for d in denominators]
    
    # Split
    n1 = n // 2
    n2 = n - n1
    
    w1 = weights[:n1]
    w2 = weights[n1:]
    
    # Generate sums and masks for part 1
    size1 = 1 << n1
    sums1 = np.zeros(size1, dtype=np.int64)
    masks1 = np.zeros(size1, dtype=np.int64)
    
    for i in range(n1):
        bit = 1 << i
        w = w1[i]
        for mask in range(1 << i):
            idx = bit | mask
            sums1[idx] = sums1[mask] + w
            masks1[idx] = masks1[mask] | bit
            
    # Generate sums and masks for part 2
    size2 = 1 << n2
    sums2 = np.zeros(size2, dtype=np.int64)
    masks2 = np.zeros(size2, dtype=np.int64)
    
    for i in range(n2):
        bit = 1 << i
        w = w2[i]
        for mask in range(1 << i):
            idx = bit | mask
            sums2[idx] = sums2[mask] + w
            masks2[idx] = masks2[mask] | bit
            
    # Sort part 1
    idx1 = np.argsort(sums1)
    sums1_sorted = sums1[idx1]
    masks1_sorted = masks1[idx1]
    
    # Sort part 2
    idx2 = np.argsort(sums2)
    sums2_sorted = sums2[idx2]
    masks2_sorted = masks2[idx2]
    
    # Find valid range for sums1 (must be < L)
    # sums1_sorted is ascending
    valid_end1 = np.searchsorted(sums1_sorted, L, side='left')
    
    # Find valid range for sums2 (must be < L, but actually we need s2 < L - s1)
    # We can just use the whole array and let the pointer handle it, 
    # but starting pos at valid_end2 is safer/faster
    valid_end2 = np.searchsorted(sums2_sorted, L, side='left')
    
    best_sum = 0
    best_mask = 0
    
    # Two-pointer
    # pos points to the first element in sums2_sorted that is >= current target
    # We want the element at pos - 1
    pos = valid_end2
    
    for k in range(valid_end1):
        s1 = sums1_sorted[k]
        target = L - s1
        
        # Move pos left while sums2_sorted[pos - 1] >= target
        while pos > 0 and sums2_sorted[pos - 1] >= target:
            pos -= 1
            
        if pos > 0:
            s2 = sums2_sorted[pos - 1]
            total = s1 + s2
            if total > best_sum:
                best_sum = total
                m1 = masks1_sorted[k]
                m2 = masks2_sorted[pos - 1]
                best_mask = m1 | (m2 << n1)
                
    # Convert best_mask to indices
    indices = []
    for i in range(n):
        if (best_mask >> i) & 1:
            indices.append(i)
            
    return indices
