"""Find the longest interval in [1,1008] with no modular inverse pairs mod 1009."""
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80  # safety margin before 90s kill
    
    p = 1009
    n = 1008  # elements are 1..1008
    
    # Precompute modular inverses: inv[x] = x^{-1} mod 1009 for x in 1..1008
    inv = [0] * (p + 1)
    for x in range(1, p):
        inv[x] = pow(x, p - 2, p)
    
    # For a contiguous interval [s, s+L-1] to be valid:
    # For every x in the interval, inv[x] must NOT be in the interval.
    # Also, x itself must not be a self-inverse (inv[x] != x), because x*x % p == 1.
    # Self-inverse elements: x^2 ≡ 1 (mod 1009) => x ≡ 1 or x ≡ -1 ≡ 1008 (mod 1009)
    # So 1 and 1008 cannot be in any valid interval.
    
    # Strategy: For each possible start s, greedily extend the interval as long as valid.
    # An element x can be added to interval [s, e] (currently ending at e, adding x=e+1) iff:
    #   1. inv[x] < s or inv[x] > e+1  (i.e., inv[x] is outside [s, e+1])
    #   2. x is not a self-inverse (x != 1 and x != 1008)
    
    # But we also need: for every existing element y in [s, e], inv[y] is not x.
    # Since inv is an involution (inv[inv[x]] = x), condition 1 for x implies inv[x] not in [s,e+1].
    # And if inv[y] = x for some y in [s,e], then inv[x] = y which is in [s,e] ⊂ [s,e+1],
    # so condition 1 for x would have caught it (inv[x] = y is in [s,e+1]).
    # So checking inv[x] not in [s, e+1] is sufficient when adding x to the right.
    
    best_start = 1
    best_length = 0
    
    for s in range(1, n + 1):
        if time.time() > deadline:
            break
        
        # If s itself is a self-inverse, skip
        if s == 1 or s == 1008:
            continue
        
        # Check if inv[s] is in [s, ...] - actually we just need inv[s] not in [s, s+L-1]
        # Let's greedily extend
        e = s
        # Check s is valid: inv[s] must not equal s (already checked) and we just have {s}
        # For a single element {s}, we need s*s % p != 1, which means s != 1 and s != 1008.
        
        valid = True
        # Extend from s
        current_end = s
        # We need to track all elements in the interval to check their inverses
        # But since it's contiguous, we can just check: for each new element x, inv[x] not in [s, x]
        # And for each existing element y in [s, x-1], inv[y] not in [s, x]
        # The second condition is equivalent to: for all y in [s, x-1], inv[y] != x
        # Which is equivalent to: x != inv[y] for all y in [s, x-1]
        # Which is equivalent to: inv[x] != y for all y in [s, x-1]
        # Which is equivalent to: inv[x] not in [s, x-1]
        # Combined with inv[x] != x (self-inverse check), we need inv[x] not in [s, x]
        
        for x in range(s, n + 1):
            if x == 1 or x == 1008:
                valid = False
                break
            ix = inv[x]
            # ix must not be in [s, x]
            if ix >= s and ix <= x:
                valid = False
                break
            current_end = x
        
        length = current_end - s + 1 if valid else (current_end - s)
        # If we broke at x, the valid interval is [s, x-1], so length = x - s
        # If we completed, length = n - s + 1
        if not valid:
            length = x - s  # x is the first invalid one, so valid up to x-1
        else:
            length = n - s + 1
        
        if length > best_length:
            best_length = length
            best_start = s
    
    if best_length <= 0:
        # Fallback: find any valid single element
        for x in range(1, n + 1):
            if x != 1 and x != 1008:
                best_start = x
                best_length = 1
                break
    
    return [best_start, best_length]
