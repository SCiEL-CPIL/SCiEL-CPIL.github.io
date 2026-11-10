import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin out of 90s
    
    N = 10**7
    
    # Step 1: Compute sigma(n) for all n up to N using vectorized sieve
    # sigma[n] = sum of all positive divisors of n
    # Sieve approach: for each d from 1 to N, add d to all multiples of d
    # This is O(N log N) but vectorized with NumPy it's fast
    
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    for d in range(1, N + 1):
        sigma[d::d] += d
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return [int(11), int(6)]
    
    # Step 2: Group numbers by their sigma value
    # Use numpy to build groups efficiently
    # First, get all n from 1 to N and their sigma values
    n_vals = np.arange(1, N + 1, dtype=np.int64)
    s_vals = sigma[1:]  # sigma[1] through sigma[N]
    
    # Sort by sigma value to group
    sort_idx = np.argsort(s_vals, kind='stable')
    s_sorted = s_vals[sort_idx]
    n_sorted = n_vals[sort_idx]
    
    # Find boundaries where sigma value changes
    # diff > 0 means a new group starts
    changes = np.concatenate(([True], s_sorted[1:] != s_sorted[:-1]))
    group_starts = np.where(changes)[0]
    group_ends = np.concatenate((group_starts[1:], [len(s_sorted)]))
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return [int(11), int(6)]
    
    # Step 3: For each group with >= 2 elements, find best pair using two-pointer
    # We want to minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # For a > b case (a/b > 1): minimize |2a - 3b| / (2b)
    # For b > a case (a/b < 1): this gives a/b < 1, far from 1.5, so less promising
    # But we should check both orientations.
    # Actually, since we can assign a and b freely, for each unordered pair {x, y}
    # we can set a=max(x,y), b=min(x,y) or vice versa.
    # a/b close to 1.5 means a ≈ 1.5b, so a > b.
    # So we want pairs where the larger is about 1.5x the smaller.
    
    best_a = 11
    best_b = 6
    # best_err as exact rational: |2a - 3b| / (2b)
    # For (11,6): |22-18|/(12) = 4/12 = 1/3
    best_num = 1  # numerator of error
    best_den = 3  # denominator of error
    # error = best_num / best_den = 1/3
    
    num_groups = len(group_starts)
    
    # Process groups in order of decreasing size first (larger groups more likely to have good pairs)
    # But actually, let's just process all groups and use a wider search window
    
    for gi in range(num_groups):
        gs = group_starts[gi]
        ge = group_ends[gi]
        k = ge - gs
        if k < 2:
            continue
        
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # Extract the group's numbers (already sorted since n_sorted is sorted by value)
        group = n_sorted[gs:ge]
        # group is sorted in increasing order
        
        # Two-pointer technique:
        # We want to find (a, b) with a > b in the group minimizing |2a - 3b| / (2b)
        # For each b (smaller element), the ideal a is 1.5*b.
        # Since group is sorted, for each b, we can binary search for 1.5*b.
        # But two-pointer: as b increases, 1.5*b increases, so the target a also increases.
        
        i = 0
        # Process from larger b values first (they give smaller error for same |2a-3b|)
        for j in range(k - 1, -1, -1):
            b_val = int(group[j])
            target_3b = 3 * b_val
            
            # Advance i so that 2*group[i] >= target_3b
            while i < k and 2 * int(group[i]) < target_3b:
                i += 1
            
            # Check a wider window around the target position
            # The optimal a is where 2*a ≈ 3*b, i.e., a ≈ 1.5*b
            # Check positions i-3 through i+3 to be thorough
            for offset in range(-3, 4):
                ii = i + offset
                if 0 <= ii < k and ii != j:
                    a_val = int(group[ii])
                    b_val2 = b_val
                    # Ensure a > b for the ratio to be near 1.5
                    if a_val <= b_val2:
                        continue
                    # Compute error = |2*a_val - 3*b_val2| / (2*b_val2)
                    diff = 2 * a_val - 3 * b_val2
                    if diff == 0:
                        continue  # exact 3/2, invalid
                    abs_diff = abs(diff)
                    den = 2 * b_val2
                    # Compare abs_diff/den with best_num/best_den
                    # abs_diff * best_den < best_num * den
                    if abs_diff * best_den < best_num * den:
                        best_num = abs_diff
                        best_den = den
                        best_a = a_val
                        best_b = b_val2
    
    return [int(best_a), int(best_b)]
