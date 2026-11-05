import math

def construct():
    n = 255
    # Find primes that divide n
    # 255 = 3 * 5 * 17
    primes = [3, 5, 17]
    
    best_set = []
    max_size = 0
    
    for p in primes:
        # Generate all multiples of p in [1, n]
        # These sets are valid because gcd(k*p, m*p) = p * gcd(k, m) >= p > 1
        # And n is a multiple of p, so n is included.
        candidates = list(range(p, n + 1, p))
        
        if len(candidates) > max_size:
            max_size = len(candidates)
            best_set = candidates
            
    # Fallback if something goes wrong (should not happen for n=255)
    if not best_set:
        return [255]
        
    return best_set
