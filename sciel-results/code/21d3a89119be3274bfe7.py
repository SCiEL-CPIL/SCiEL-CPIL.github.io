"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 80.0  # safety margin before 90s kill
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes)
    
    def count_distinct_small_primes(n, window=24):
        """Count distinct small primes in n+1..n+window"""
        seen = set()
        for i in range(1, window + 1):
            x = n + i
            for p in primes:
                if x % p == 0:
                    seen.add(p)
                    while x % p == 0:
                        x //= p
                    if x == 1:
                        break
        return len(seen)
    
    best_n = 1000
    best_count = count_distinct_small_primes(1000)
    
    # Phase 1: Quick scan with step to find good regions
    # Try every 100th value first for a rough landscape
    n = 1000
    while n <= 1000000:
        if time.time() > deadline - 10:
            break
        c = count_distinct_small_primes(n)
        if c < best_count:
            best_count = c
            best_n = n
        n += 100
    
    # Phase 2: Fine-grained search around best regions
    # Search in windows around the best found
    if time.time() < deadline - 5:
        search_range = 500
        center = best_n
        for delta in range(-search_range, search_range + 1):
            if time.time() > deadline - 5:
                break
            n = center + delta
            if n < 1000 or n > 1000000:
                continue
            c = count_distinct_small_primes(n)
            if c < best_count:
                best_count = c
                best_n = n
    
    # Phase 3: Broader fine-grained search if time remains
    if time.time() < deadline - 2:
        # Scan more broadly with step 1 in promising ranges
        # Try ranges around multiples of primorial-like numbers
        # The product of primes < 24 is huge, but let's try strategic ranges
        step = 1
        n = 1000
        while n <= 1000000:
            if time.time() > deadline - 2:
                break
            c = count_distinct_small_primes(n)
            if c < best_count:
                best_count = c
                best_n = n
            n += step
    
    return best_n
# EVOLVE-BLOCK-END
