import math

def construct():
    n = 255
    # Prime factors of 255 are 3, 5, 17
    primes = [3, 5, 17]
    best_set = []
    best_size = 0
    
    for p in primes:
        # Generate multiples of p in [1, n]
        # This set contains n (since p divides n)
        # All pairs share factor p, so gcd > 1
        candidates = list(range(p, n + 1, p))
        # Check if n is in candidates (it should be)
        if n in candidates:
            if len(candidates) > best_size:
                best_size = len(candidates)
                best_set = candidates
                
    # Fallback if something goes wrong (should not happen)
    if not best_set:
        best_set = [n]
        
    return best_set
