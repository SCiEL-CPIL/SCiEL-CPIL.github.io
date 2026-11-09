import bisect
from fractions import Fraction
import time

def construct():
    denominators = list(range(2, 49))
    n = len(denominators)
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 45.0  # 15s margin
    
    # Split into two halves
    mid = n // 2
    d1 = denominators[:mid]
    d2 = denominators[mid:]
    
    # Generate subset sums for d1 using floats
    # We store (float_sum, mask)
    sums1 = [(0.0, 0)]
    for i, d in enumerate(d1):
        val = 1.0 / d
        new_sums = []
        for s, m in sums1:
            new_s = s + val
            if new_s < 1.0:
                new_sums.append((new_s, m | (1 << i)))
        sums1.extend(new_sums)
        
    # Generate subset sums for d2 using floats
    sums2 = [(0.0, 0)]
    for i, d in enumerate(d2):
        val = 1.0 / d
        new_sums = []
        for s, m in sums2:
            new_s = s + val
            if new_s < 1.0:
                new_sums.append((new_s, m | (1 << i)))
        sums2.extend(new_sums)
        
    # Sort sums2 by value
    sums2.sort(key=lambda x: x[0])
    # Extract values for binary search
    vals2 = [s for s, m in sums2]
    
    # We will collect a list of candidate (deficit_float, mask1, mask2)
    # To be safe against float errors, we can keep the top few
    candidates = []
    
    for s1, m1 in sums1:
        if time.time() > deadline:
            break
            
        # We need s2 < 1.0 - s1
        target = 1.0 - s1
        # Find rightmost index in vals2 such that vals2[i] < target
        # bisect_left gives first index >= target
        idx = bisect.bisect_left(vals2, target)
        
        if idx > 0:
            # The best candidate is at idx - 1
            s2 = vals2[idx - 1]
            m2 = sums2[idx - 1][1]
            
            total_sum = s1 + s2
            deficit = 1.0 - total_sum
            
            candidates.append((deficit, m1, m2))
            
    # Sort candidates by deficit (ascending)
    candidates.sort(key=lambda x: x[0])
    
    best_deficit = None
    best_indices = None
    
    # Verify top candidates with Fraction
    for deficit, m1, m2 in candidates[:10]: # Check top 10
        # Reconstruct indices
        indices = []
        for i in range(len(d1)):
            if m1 & (1 << i):
                indices.append(i)
        for i in range(len(d2)):
            if m2 & (1 << i):
                indices.append(mid + i)
                
        if not indices:
            continue
            
        # Compute exact sum
        exact_sum = Fraction(0)
        for idx in indices:
            exact_sum += Fraction(1, denominators[idx])
            
        if exact_sum < 1:
            if best_deficit is None or (1 - exact_sum) < best_deficit:
                best_deficit = 1 - exact_sum
                best_indices = indices
        if time.time() > deadline:
            break
            
    if best_indices is None:
        # Fallback: return a known valid small set
        # e.g. just [0] (1/2)
        best_indices = [0]
        
    return best_indices
