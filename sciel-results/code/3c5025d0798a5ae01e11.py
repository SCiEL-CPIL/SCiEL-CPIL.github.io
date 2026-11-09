"""Approximate 3/2 ratio within a divisor-sum fiber."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from collections import defaultdict
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 78  # safety margin before 90s kill
    
    MAX_N = 10_000_000
    TARGET_NUM = 3
    TARGET_DEN = 2
    
    # Compute sigma(n) for all n up to MAX_N using sieve
    sigma = np.zeros(MAX_N + 1, dtype=np.int64)
    for d in range(1, MAX_N + 1):
        sigma[d::d] += d
        # Periodic time check
        if d % 500_000 == 0:
            if time.time() > deadline - 10:
                break
    
    # Check if we completed the sieve
    if time.time() > deadline - 8:
        # Fallback: return the seed answer
        return [11, 6]
    
    # Build groups: sigma value -> sorted list of n
    # To save memory and time, we only store n values
    groups = defaultdict(list)
    for n in range(1, MAX_N + 1):
        groups[sigma[n]].append(n)
    
    if time.time() > deadline - 5:
        return [11, 6]
    
    # For each group, find pairs (a, b) minimizing |a/b - 3/2|
    # a/b ≈ 3/2 means a ≈ 3b/2, so 2a ≈ 3b
    # We want to minimize |2a - 3b| / (2b) = |2a/b - 3| / 2
    # Equivalently minimize |2a - 3b| / b (since we divide by 2b, minimizing |2a-3b|/b is same)
    # Actually: |a/b - 3/2| = |2a - 3b| / (2b)
    # So we want to minimize |2a - 3b| / b
    
    best_error = None
    best_pair = (11, 6)
    
    for sig_val, members in groups.items():
        if time.time() > deadline - 3:
            break
        
        if len(members) < 2:
            continue
        
        members.sort()
        n_members = len(members)
        
        # For each b in members, find a in members closest to 3b/2
        # Since members is sorted, we can use binary search
        for i in range(n_members):
            b = members[i]
            target_a = 3 * b / 2.0
            
            # Binary search for closest to target_a
            # We need a such that 2a ≈ 3b, i.e., a ≈ 3b/2
            # Use bisect
            idx = np.searchsorted(members, target_a)
            
            candidates = []
            if idx < n_members:
                candidates.append(members[idx])
            if idx > 0:
                candidates.append(members[idx - 1])
            
            for a in candidates:
                if a == b:
                    continue
                # Compute error: |a/b - 3/2| = |2a - 3b| / (2b)
                num = abs(2 * a - 3 * b)
                den = 2 * b
                if num == 0:
                    # Exact match - but task says raw value of exactly 0 is invalid
                    continue
                error = num / den
                
                if best_error is None or error < best_error:
                    best_error = error
                    best_pair = (a, b)
                    
                    # Also check the reverse: could b/a be closer?
                    # Actually we're checking all pairs where a is the "numerator" and b is the "denominator"
                    # But we should also consider swapping: what if we set a' = b and b' = a?
                    # |b/a - 3/2| = |2b - 3a| / (2a)
                    # This is a different pair, and it will be found when we process that b value
                    # So we don't need to explicitly check both directions here.
    
    if best_error is None:
        return [11, 6]
    
    a, b = best_pair
    # Ensure they are Python ints
    a = int(a)
    b = int(b)
    
    # Validate
    if not isinstance(a, int) or not isinstance(b, int):
        return [11, 6]
    if a <= 0 or b <= 0 or a > MAX_N or b > MAX_N:
        return [11, 6]
    
    # Double-check sigma(a) == sigma(b)
    if sigma[a] != sigma[b]:
        return [11, 6]
    
    return [a, b]

# EVOLVE-BLOCK-END
