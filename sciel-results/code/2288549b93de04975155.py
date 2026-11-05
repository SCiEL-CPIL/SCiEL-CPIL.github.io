import time
import math

def construct():
    n = 256
    deadline = time.time() + 80  # 80s budget, 10s safety margin
    
    # Start with the known valid set {171, ..., 256}
    S = list(range(171, n + 1))
    S_set = set(S)
    
    # Precompute for fast divisibility checks
    # For each a in 1..n, precompute which sums b+c are divisible by a
    # We'll use this to speed up candidate validation
    
    def is_valid_with_candidate(S_list, m):
        """Check if adding m to S_list maintains the property using optimized checks."""
        S_sorted = sorted(S_list)
        
        # Case 1: m is the smallest (a), so m < b <= c for b,c in S_list
        # Need m does not divide b+c for all b,c in S_list with b > m, c >= b
        elements_greater = [x for x in S_sorted if x > m]
        for i, b in enumerate(elements_greater):
            for c in elements_greater[i:]:
                if (b + c) % m == 0:
                    return False
        
        # Case 2: m is the middle (b), so a < m <= c for a,c in S_list
        # Need a does not divide m+c for all a in S_list with a < m, c in S_list with c >= m
        elements_less = [x for x in S_sorted if x < m]
        elements_geq_m = [x for x in S_sorted if x >= m]
        for a in elements_less:
            for c in elements_geq_m:
                if (m + c) % a == 0:
                    return False
        
        # Case 3: m is the largest (c), so a < b <= m for a,b in S_list
        # Need a does not divide b+m for all a,b in S_list with a < b <= m
        elements_leq_m = [x for x in S_sorted if x <= m]
        for i, a in enumerate(elements_leq_m):
            for j in range(i + 1, len(elements_leq_m)):
                b = elements_leq_m[j]
                if (b + m) % a == 0:
                    return False
        
        return True
    
    # Try adding elements from 170 down to 1
    for m in range(170, 0, -1):
        if time.time() > deadline:
            break
        if is_valid_with_candidate(S, m):
            S.append(m)
            S_set.add(m)
    
    # Second pass: try adding elements again after removals might have created space
    # First, identify which elements are "blocking" others
    # We'll do a greedy removal and re-addition to potentially improve
    
    # Try to remove elements that block many others, then re-add
    S_sorted = sorted(S)
    
    # For each element in S, count how many potential additions it blocks
    # This is expensive, so we'll do a simpler heuristic:
    # Try removing small elements first (they block more due to divisibility)
    
    removal_candidates = []
    for i, elem in enumerate(S_sorted):
        if elem <= 170:  # Only consider removing added elements
            removal_candidates.append((elem, i))
    
    # Sort by value ascending (smaller elements block more)
    removal_candidates.sort()
    
    improved = False
    for elem, idx in removal_candidates:
        if time.time() > deadline:
            break
        if len(S) <= 86:  # Don't go below baseline
            break
            
        # Temporarily remove elem
        S.remove(elem)
        S_set.discard(elem)
        
        # Try to add back some elements we couldn't add before
        added_any = False
        for m in range(170, 0, -1):
            if time.time() > deadline:
                break
            if m in S_set:
                continue
            if is_valid_with_candidate(S, m):
                S.append(m)
                S_set.add(m)
                added_any = True
        
        if not added_any or len(S) <= 86:
            # Restore
            S.append(elem)
            S_set.add(elem)
    
    # Final verification
    S_sorted = sorted(S)
    valid = True
    for i, a in enumerate(S_sorted):
        for j in range(i + 1, len(S_sorted)):
            b = S_sorted[j]
            for c_val in S_sorted[j:]:
                c = c_val
                if (b + c) % a == 0:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback
        return list(range(171, 257))
    
    return S
