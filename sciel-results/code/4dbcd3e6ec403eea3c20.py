import time

def construct():
    deadline = time.time() + 85  # safety margin before 90s kill
    
    # Primes strictly below 24
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    small_prime_set = set(small_primes)
    
    def extract_small_factor(x):
        """Extract the complete factor of x composed of primes strictly below 24."""
        result = 1
        for p in small_primes:
            while x % p == 0:
                result *= p
                x //= p
        return result
    
    # Precompute for a sliding window approach
    # We want to minimize distinct values of extract_small_factor(n+i) for i=1..24
    
    best_score = float('inf')
    best_n = 1000
    
    # First pass: scan in chunks for speed
    # Use a faster approach: precompute extract_small_factor for a range
    
    # Let's do a direct scan but with early termination if possible
    # For speed, we can use the fact that if we already have 24 distinct values, 
    # we can't do better, but we need the actual minimum.
    
    # Let's scan from 1000 to 1000000
    # To speed up, precompute the small-factor for a window of 1000 numbers at a time
    
    import numpy as np
    
    def batch_extract(arr):
        """Extract small prime factors for an array of numbers."""
        result = np.ones(len(arr), dtype=np.int64)
        for p in small_primes:
            mask = (arr % p == 0)
            while np.any(mask):
                result[mask] *= p
                arr = arr.copy()  # need to modify
                arr[mask] //= p
                mask = (arr % p == 0)
        return result
    
    # Actually, let's just do a simple loop but optimize the inner check
    # The range is 1000 to 1000000, so 999000 candidates, each needing 24 extractions
    
    # Faster: precompute extract_small_factor for all numbers in [1001, 1000024]
    # That's about 999024 numbers. Each extraction is fast (at most 9 primes to check).
    
    max_val = 1000000 + 24
    # Precompute array
    small_factors = [0] * (max_val + 1)
    
    for x in range(1, max_val + 1):
        xf = x
        sf = 1
        for p in small_primes:
            while xf % p == 0:
                sf *= p
                xf //= p
        small_factors[x] = sf
    
    # Now scan for best n
    for n in range(1000, 1000001):
        if time.time() > deadline:
            break
        
        # Get the 24 extracted factors
        factors = [small_factors[n + i] for i in range(1, 25)]
        distinct_count = len(set(factors))
        
        if distinct_count < best_score:
            best_score = distinct_count
            best_n = n
            if best_score == 1:
                break  # Can't do better than 1
    
    return best_n
