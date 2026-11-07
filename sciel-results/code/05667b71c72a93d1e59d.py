import time
import numpy as np
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = 85.0  # safety margin before 90s kill
    
    # We'll compute sigma up to N, then group by sigma value
    # To fit in time, we'll use a smaller N if needed
    # Let's try N = 5_000_000 first, and if time permits, go higher
    
    N = 5_000_000
    
    # Compute sigma(n) for all n up to N using a sieve
    sigma = np.zeros(N + 1, dtype=np.int64)
    
    # Sieve: for each i, add i to sigma[j] for all j that are multiples of i
    for i in range(1, N + 1):
        if time.time() - start_time > deadline - 15:
            # Fallback to a known pair
            return [11, 6]
        sigma[i::i] += i
    
    if time.time() - start_time > deadline - 5:
        return [11, 6]
    
    # Group numbers by their sigma value
    sigma_groups = defaultdict(list)
    for n in range(1, N + 1):
        sigma_groups[sigma[n]].append(n)
    
    if time.time() - start_time > deadline - 3:
        return [11, 6]
    
    best_error = float('inf')
    best_a = 11
    best_b = 6
    
    # For each sigma group with at least 2 elements, find pairs (a,b)
    # that minimize |a/b - 3/2| = |2a - 3b| / (2b)
    # For a given b, we want a as close to 3b/2 as possible
    # Since a and b are in the same group, we can use binary search
    
    for s, group in sigma_groups.items():
        if len(group) < 2:
            continue
        if time.time() - start_time > deadline - 2:
            break
        
        # Sort the group for binary search
        group.sort()
        n_group = len(group)
        
        # For each b in group, find a in group closest to 3b/2
        for i in range(n_group):
            b = group[i]
            target = 3 * b / 2.0
            
            # Binary search for the element closest to target
            # We want to find the position where target would be inserted
            lo, hi = 0, n_group - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if group[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid
            
            # Check lo and lo-1 (if they exist)
            candidates = set()
            if lo < n_group:
                candidates.add(group[lo])
            if lo - 1 >= 0:
                candidates.add(group[lo - 1])
            # Also check a few neighbors in case the closest isn't exactly at the boundary
            for offset in [-2, -1, 1, 2]:
                idx = lo + offset
                if 0 <= idx < n_group:
                    candidates.add(group[idx])
            
            for a in candidates:
                if a == b:
                    continue
                # Check if ratio is exactly 3/2 (invalid)
                if 2 * a == 3 * b:
                    continue
                
                num = abs(2 * a - 3 * b)
                den = 2 * b
                err = num / den
                
                if err < best_error:
                    best_error = err
                    best_a = a
                    best_b = b
                    
    return [int(best_a), int(best_b)]
