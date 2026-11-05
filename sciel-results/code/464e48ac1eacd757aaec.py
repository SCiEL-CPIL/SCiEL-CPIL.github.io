import time
from math import gcd
from functools import reduce
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55.0

    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Calculate LCM
    def lcm(a, b):
        return a * b // gcd(a, b)
    
    L = reduce(lcm, denominators)
    
    # Values
    V = [L // d for d in denominators]
    
    # Split
    n1 = 23
    n2 = n - n1
    V1 = V[:n1]
    V2 = V[n1:]
    
    # Generate sums and masks for V1
    def gen_sums(vals):
        sums = [0]
        masks = [0]
        for i, v in enumerate(vals):
            # Optimization: check time
            if time.time() > deadline:
                return sums, masks
            new_sums = [s + v for s in sums]
            new_masks = [m | (1 << i) for m in masks]
            sums.extend(new_sums)
            masks.extend(new_masks)
        return sums, masks

    sums1, masks1 = gen_sums(V1)
    
    if time.time() > deadline:
        # Fallback to a simple greedy or previous result if timeout
        return [0] 

    sums2, masks2 = gen_sums(V2)
    
    if time.time() > deadline:
        return [0]

    # Sort sums2 and masks2 together by sum
    # Create list of tuples (sum, mask)
    pairs2 = list(zip(sums2, masks2))
    pairs2.sort(key=lambda x: x[0])
    
    # Extract sorted sums for bisect
    sorted_sums2 = [p[0] for p in pairs2]
    
    best_total = 0
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
        # bisect_right returns insertion point. We want the element before it if it exists.
        # bisect_right(sorted_sums2, target) gives index where target would be inserted to keep order.
        # So index - 1 is the largest element <= target.
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
        indices = [0]
        
    return indices

if __name__ == "__main__":
    res = construct()
