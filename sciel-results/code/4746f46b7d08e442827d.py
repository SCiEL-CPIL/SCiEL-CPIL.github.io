"""Optimized odd covering system search using integer bitset greedy maximum coverage."""

BASE = 51975
# 51975 = 3^3 * 5^2 * 7 * 11

def _get_divisors(n):
    """Get all divisors of n greater than 1."""
    divs = []
    for d in range(2, n + 1):
        if n % d == 0:
            divs.append(d)
    return divs

def construct():
    base = BASE
    # Precompute all divisors > 1 of 51975
    divisors = _get_divisors(base)
    
    # Precompute masks for each (m, r) pair
    # mask is an integer where bit i is 1 if i % m == r
    # We store masks as a list of (m, r, mask) tuples
    candidates = []
    for m in divisors:
        for r in range(m):
            mask = 0
            # Set bits for all i in [0, base) such that i % m == r
            # i = r, r+m, r+2m, ...
            i = r
            while i < base:
                mask |= (1 << i)
                i += m
            candidates.append((m, r, mask))
    
    # Sort candidates by potential coverage (smaller m covers more)
    # This might help if we break ties or if we want to prioritize, 
    # but the greedy loop will check all anyway.
    candidates.sort(key=lambda x: x[0])
    
    used_m = set()
    uncovered = (1 << base) - 1  # All bits set to 1
    classes = []
    
    # Greedy: repeatedly pick the (m, r) that covers the most uncovered residues
    for _ in range(len(divisors)):
        best_gain = -1
        best_pair = None
        best_mask = 0
        
        for m, r, mask in candidates:
            if m in used_m:
                continue
            
            # Count uncovered residues in this class
            intersection = mask & uncovered
            gain = intersection.bit_count()
            
            if gain > best_gain:
                best_gain = gain
                best_pair = (m, r)
                best_mask = mask
        
        if best_pair is None or best_gain <= 0:
            break
        
        m, r = best_pair
        used_m.add(m)
        classes.append([m, r])
        
        # Update uncovered set
        uncovered &= ~best_mask
        
        # Optional: if uncovered becomes 0, we can break early
        if uncovered == 0:
            break
    
    return classes
