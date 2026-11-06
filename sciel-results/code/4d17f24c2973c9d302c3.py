import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = 50  # 50 seconds safety margin
    
    target = Fraction(5, 31)
    
    # Search for the minimum largest denominator
    # Based on analysis, M is likely small (< 100)
    for M in range(7, 1000):
        if time.time() > deadline:
            break
            
        # Find a set of distinct denominators in [7, M] summing to target
        # Use backtracking
        found = _find_expansion(target, 7, M, [])
        if found:
            return found
            
    # Fallback to greedy if not found in time (should not happen for small M)
    r = target
    values = []
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values

def _find_expansion(rem, min_d, max_d, current):
    if rem == 0:
        return current
    if rem < 0 or not current: # Should not happen with correct logic
        return None
        
    # Pruning: 
    # 1. If rem is smaller than 1/max_d, we can't add any more terms (since all terms >= 1/max_d)
    #    Wait, we can add terms if we haven't hit max_d? No, we are picking from min_d to max_d.
    #    If rem < 1/max_d, we can't form rem with any denominator <= max_d.
    if rem < Fraction(1, max_d):
        return None
        
    # 2. If rem is larger than sum of all possible remaining terms, return None
    #    Max sum is sum(1/i for i in range(min_d, max_d+1))
    #    This is expensive to compute repeatedly.
    #    Simpler bound: If rem > sum(1/i for i in range(min_d, max_d+1)), fail.
    #    For small M, this is fast.
    
    # Heuristic: Try larger denominators first to keep sum small?
    # Or smaller?
    # We want to hit exactly rem.
    # If we pick small d, rem decreases a lot.
    # If we pick large d, rem decreases a little.
    
    # Iterate d from max_d down to min_d?
    # If we iterate down, we might find a solution with large denominators first?
    # But we want ANY solution for this M.
    
    for d in range(max_d, min_d - 1, -1):
        if d in current:
            continue
        if rem < Fraction(1, d):
            continue # If rem is too small for 1/d, smaller d will be too big
            # Actually if rem < 1/d, then for any d' < d, 1/d' > 1/d > rem.
            # So we can break? No, we are iterating downwards.
            # If d is small, 1/d is large.
            # We want 1/d <= rem.
            # So d >= 1/rem.
            # If d < 1/rem, we can't use it.
            # Since we iterate max_d down to min_d:
            # If d is too small (1/d > rem), we should skip it.
            # But all smaller d will also have 1/d > rem.
            # So we can break?
            # No, we are going down. d decreases. 1/d increases.
            # If 1/d > rem, then for all d' < d, 1/d' > 1/d > rem.
            # So we can break the loop.
            break
            
        new_rem = rem - Fraction(1, d)
        if new_rem == 0:
            return current + [d]
            
        # Recurse
        # Next min_d is d+1? No, denominators must be distinct.
        # We can pick any d' > d? Or any d' != d?
        # Usually Egyptian fractions have increasing denominators.
        # If we assume increasing, next min is d+1.
        # But we are iterating d downwards.
        # If we pick d, the next term must be > d?
        # If we enforce d1 < d2 < ..., then if we pick d, next must be > d.
        # But we are iterating from max down.
        # If we pick d, we can only pick from d+1 to max_d?
        # No, we already considered larger denominators in previous iterations?
        # No, this is a search tree.
        # If we enforce increasing order, we should iterate d from min_d to max_d.
        
    return None

# Revised search with increasing order
def _find_expansion_inc(rem, min_d, max_d, current):
    if rem == 0:
        return current
    if rem < 0:
        return None
    if min_d > max_d:
        return None
        
    # Pruning
    # Min possible sum of k terms starting from min_d is sum(1/i for i in min_d..max_d)
    # If rem > that sum, fail.
    # Max possible sum of 1 term is 1/min_d.
    # If rem < 1/max_d, fail (can't make rem with any term <= max_d)
    if rem < Fraction(1, max_d):
        return None
        
    # Iterate d from min_d to max_d
    # To find solution fast, maybe try d such that 1/d is close to rem?
    # d ~ 1/rem
    # Let's just iterate.
    
    for d in range(min_d, max_d + 1):
        if d in current:
            continue
        if Fraction(1, d) > rem:
            continue # Too big, try smaller d? No, d increases, 1/d decreases.
            # If 1/d > rem, then for all d' < d, 1/d' > 1/d > rem.
            # So we should have started with larger d?
            # If we iterate min to max, and 1/d > rem, we can't use d.
            # But smaller d (which we already passed?) No, we start at min_d.
            # If min_d is small, 1/min_d is large.
            # If 1/min_d > rem, we can't use min_d.
            # We must skip all d such that 1/d > rem.
            # 1/d <= rem => d >= 1/rem.
            # So we can start from max(min_d, ceil(1/rem)).
            
        # Optimization: Start from ceil(1/rem)
        
        new_rem = rem - Fraction(1, d)
        if new_rem == 0:
            return current + [d]
            
        res = _find_expansion_inc(new_rem, d + 1, max_d, current + [d])
        if res:
            return res
            
    return None

def construct():
    start_time = time.time()
    deadline = 50
    
    target = Fraction(5, 31)
    
    for M in range(7, 1000):
        if time.time() > deadline:
            break
            
        # We need to find a subset of {7..M} summing to target.
        # Using increasing order search.
        found = _find_expansion_inc(target, 7, M, [])
        if found:
            return found
            
    # Fallback
    r = target
    values = []
    while r:
        d = (r.denominator + r.numerator - 1) // r.numerator
        values.append(d)
        r -= Fraction(1, d)
    return values
