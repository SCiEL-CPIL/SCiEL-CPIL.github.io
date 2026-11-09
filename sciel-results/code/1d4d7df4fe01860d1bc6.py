"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80  # Safety margin before 90s limit
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes)
    
    def small_prime_mask(x):
        """Return a bitmask of which small primes divide x."""
        mask = 0
        for i, p in enumerate(primes):
            if x % p == 0:
                mask |= (1 << i)
        return mask
    
    # Precompute masks for a range of numbers
    # We need n+1 to n+24, so we need masks for numbers up to 1000024
    max_val = 1000024
    masks = [0] * (max_val + 1)
    
    for i in range(1, max_val + 1):
        m = 0
        for j, p in enumerate(primes):
            if i % p == 0:
                m |= (1 << j)
        masks[i] = m
    
    def count_distinct_primes_in_window(n):
        """Count distinct small primes in the window n+1 to n+24."""
        union_mask = 0
        for i in range(1, 25):
            union_mask |= masks[n + i]
        return bin(union_mask).count('1')
    
    best_n = 1000
    best_count = count_distinct_primes_in_window(best_n)
    
    # Try to find better solutions
    # First, do a coarse scan with stride
    stride = 1
    n = 1000
    while n <= 1000000:
        if time.time() > deadline:
            break
        
        # Check a block of n values
        for k in range(n, min(n + 100, 1000001)):
            if time.time() > deadline:
                break
            cnt = count_distinct_primes_in_window(k)
            if cnt < best_count:
                best_count = cnt
                best_n = k
                if best_count == 1:
                    return best_n
                if best_count == 2:
                    # Found a good one, do a more thorough local search
                    break
        n += 100
    
    # If we found count 2, do a more detailed search around promising areas
    if best_count <= 2:
        # Search more carefully
        for k in range(1000, 1000001):
            if time.time() > deadline:
                break
            cnt = count_distinct_primes_in_window(k)
            if cnt < best_count:
                best_count = cnt
                best_n = k
                if best_count == 1:
                    return best_n
    
    return best_n
# EVOLVE-BLOCK-END
