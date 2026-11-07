"""Improved program: compute sigma via sieve, group by sigma, find min |a/b - 3/2|."""
import time
import math
from fractions import Fraction

_MAX = 10_000_000
_TARGET = Fraction(3, 2)

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin
    
    N = _MAX
    target_num, target_den = 3, 2  # a/b = 3/2 means 2a = 3b
    
    # Step 1: Compute sigma(n) for all n <= N using sieve
    sigma = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1, i):
            sigma[j] += i
        if time.time() > deadline:
            break
    
    # Step 2: Group indices by sigma value
    # Only keep groups with >= 2 elements
    groups = {}
    for n in range(1, N + 1):
        s = sigma[n]
        if s in groups:
            groups[s].append(n)
        else:
            groups[s] = [n]
    
    if time.time() > deadline:
        # Fallback: return a known valid pair
        return [11, 6]
    
    # Step 3: For each group, find pair (a, b) minimizing |a/b - 3/2|
    # a/b - 3/2 = (2a - 3b) / (2b)
    # Minimize |2a - 3b| / (2b)
    # For a fixed b, we want a closest to 3b/2
    # So for each b, search for a in the group closest to 3b/2
    
    best_error = Fraction(1, 1)  # very large
    best_pair = (11, 6)
    
    for s, members in groups.items():
        if len(members) < 2:
            continue
        if time.time() > deadline:
            break
        
        # Sort members for binary search
        members_sorted = sorted(members)
        
        for b in members_sorted:
            if time.time() > deadline:
                break
            
            # We want a closest to 3*b/2
            target_a = 3 * b // 2
            
            # Binary search for closest a to target_a
            lo, hi = 0, len(members_sorted) - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if members_sorted[mid] < target_a:
                    lo = mid + 1
                else:
                    hi = mid
            
            # Check lo and lo-1
            candidates = set()
            if lo < len(members_sorted):
                candidates.add(members_sorted[lo])
            if lo - 1 >= 0:
                candidates.add(members_sorted[lo - 1])
            
            for a in candidates:
                if a == b:
                    continue
                err = abs(Fraction(2 * a, 2 * b) - _TARGET)
                # Actually: |a/b - 3/2| = |2a - 3b| / (2b)
                err = Fraction(abs(2 * a - 3 * b), 2 * b)
                if err < best_error:
                    best_error = err
                    best_pair = (a, b)
    
    # Verify result
    a, b = best_pair
    if sigma[a] != sigma[b]:
        return [11, 6]
    if a <= 0 or b <= 0 or a > _MAX or b > _MAX:
        return [11, 6]
    if a == b:
        return [11, 6]
    
    return [a, b]
