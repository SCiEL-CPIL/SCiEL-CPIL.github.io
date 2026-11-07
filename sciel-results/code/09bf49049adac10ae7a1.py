import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80 second deadline to be safe
    
    MAX_N = 10_000_000
    target_num, target_den = 3, 2
    
    # Compute sigma(n) for all n up to MAX_N using a sieve
    # sigma[n] = sum of all positive divisors of n
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    
    # Sieve: for each d, add d to sigma[d], sigma[2d], sigma[3d], ...
    for d in range(1, MAX_N + 1):
        for m in range(d, MAX_N + 1, d):
            sigma[m] += d
    
    # Check time
    if time.time() > deadline - 5:
        # Fallback to known pair
        return [11, 6]
    
    # Group numbers by their sigma value
    # We need to be memory-efficient. Let's use a dictionary.
    # But with 10^7 entries, this could be large. Let's try.
    
    # Actually, let's be smarter. We want pairs (a,b) with sigma[a]==sigma[b] and a/b close to 3/2.
    # For a given b, we want a ≈ 3b/2. So a is in range [3b/2 - eps, 3b/2 + eps].
    # 
    # Strategy: 
    # 1. Build a dict: sigma_val -> list of n (but this uses a lot of memory)
    # 2. Alternative: sort by sigma value, then for each group, find pairs near ratio 3/2
    
    # Let's try building the dict but be careful about memory
    # Actually, let's think about this differently.
    
    # For each b from 1 to MAX_N, compute target_a = 3*b/2
    # We want to find a in the same sigma group as b, with a close to target_a.
    # 
    # Better approach: 
    # 1. Create an array indexed by n, storing sigma[n]
    # 2. For searching, we can iterate over b, compute target_a, and check nearby values
    #    to see if any have the same sigma as b.
    # 
    # But "check nearby values" is O(1) per b if we just check a few candidates.
    # The issue is that a and b must have EXACTLY the same sigma, not just close.
    
    # Let me try a different approach:
    # Build a dictionary mapping sigma_value -> sorted list of n
    # Then for each group, for each b in the group, look for a in the group near 3b/2
    
    # Memory concern: 10^7 integers in lists... should be OK
    from collections import defaultdict
    sigma_groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        sigma_groups[sigma[n]].append(n)
    
    if time.time() > deadline - 5:
        return [11, 6]
    
    # Now search for best pair
    best_error = None
    best_pair = (11, 6)  # default
    best_a = 11
    best_b = 6
    
    # For efficiency, only consider groups with at least 2 elements
    # For each group, sort is already done (n is increasing)
    # For each b in group, target_a = 3*b/2, use binary search to find closest a in group
    
    import bisect
    
    for sigma_val, group in sigma_groups.items():
        if len(group) < 2:
            continue
        
        # group is sorted (since we appended in order of n)
        for b in group:
            target_a = 3 * b / 2.0
            # Find position where target_a would be inserted
            pos = bisect.bisect_left(group, int(target_a))
            
            # Check candidates around pos
            candidates = []
            for p in [pos - 1, pos, pos + 1]:
                if 0 <= p < len(group):
                    candidates.append(group[p])
            
            for a in candidates:
                if a == b:
                    continue
                # Compute exact error |a/b - 3/2| = |2a - 3b| / (2b)
                error_num = abs(2 * a - 3 * b)
                error_den = 2 * b
                if error_num == 0:
                    # This would be invalid, skip
                    continue
                error = error_num / error_den
                
                if best_error is None or error < best_error:
                    best_error = error
                    best_pair = (a, b)
                    best_a = a
                    best_b = b
        
        if time.time() > deadline - 5:
            break
    
    if best_error is None:
        return [11, 6]
    
    return [int(best_a), int(best_b)]
