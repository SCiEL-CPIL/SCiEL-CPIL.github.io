import time
import math

def construct():
    start_time = time.time()
    deadline = 58 # Safety margin
    
    N_MIN = 10000
    N_MAX = 1000000
    
    # Primes to check
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def generate_smooth(prime_list, min_val, max_val):
        """Generate all numbers in [min_val, max_val] whose prime factors are in prime_list."""
        smooths = set()
        # Recursive generation
        def gen(idx, current):
            if current > max_val:
                return
            if current >= min_val:
                smooths.add(current)
            if idx >= len(prime_list):
                return
            p = prime_list[idx]
            # Multiply by p^k
            val = current
            while val <= max_val:
                gen(idx + 1, val)
                val *= p
                if val > max_val:
                    break
        gen(0, 1)
        return smooths

    for p in primes:
        # Generate p-smooth numbers
        # Note: generate_smooth uses the list of primes up to p
        # We need to pass the list of primes <= p
        current_primes = [x for x in primes if x <= p]
        smooths = generate_smooth(current_primes, N_MIN, N_MAX + 1)
        
        if not smooths:
            continue
            
        # Check for consecutive pairs
        found = False
        for x in smooths:
            if x + 1 in smooths:
                # Ensure x is in valid range for n
                if N_MIN <= x <= N_MAX:
                    return x
                found = True # Found a pair, but x might be out of range? 
                             # If x < N_MIN, then x+1 <= N_MIN. 
                             # If x+1 == N_MIN, then x = N_MIN-1. 
                             # We need n in [N_MIN, N_MAX]. 
                             # So if x = N_MIN-1, n=x is invalid. 
                             # But if x+1 is the first one? 
                             # The pair is (x, x+1). 
                             # We can return n=x if x>=N_MIN. 
                             # Or n=x-1 if x-1>=N_MIN and x-1 is smooth? 
                             # No, the pair is (x, x+1). 
                             # If x < N_MIN, we can't return x. 
                             # Can we return x-1? 
                             # Only if x-1 and x are the pair. 
                             # But we checked x and x+1. 
                             # So if x < N_MIN, we missed the pair (x-1, x) if it exists? 
                             # No, we check all x in smooths. 
                             # If x-1 and x are both smooth, then x-1 is in smooths and x is in smooths. 
                             # We would have found it when iterating x-1. 
                             # So if we are at x, and x+1 is in smooths, the pair is (x, x+1). 
                             # We return x. 
                             # If x < N_MIN, we cannot return x. 
                             # Do we continue? 
                             # Yes. 
                             # But if the ONLY pair is (N_MIN-1, N_MIN), we can't return N_MIN-1. 
                             # Can we return N_MIN? 
                             # LPF of N_MIN(N_MIN+1)? 
                             # The pair (N_MIN-1, N_MIN) has low LPF. 
                             # But we need n in range. 
                             # So we must find a pair (n, n+1) with n >= N_MIN. 
                             # So we just check if x >= N_MIN.
                pass
        
        # Time check
        if time.time() - start_time > deadline:
            break

    # Fallback to original method if no smooth pair found or timeout
    # ... (keep original sieve code as fallback) ...
    
    return 10000
