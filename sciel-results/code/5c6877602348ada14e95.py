"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import isqrt

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin before 90s limit
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    num_primes = len(primes)
    
    # Precompute for each number in [1001, 1000024] the bitmask of small-prime factors
    # and the actual small part (product of prime powers)
    max_n = 1000000
    max_val = max_n + 24
    
    # For each number, compute which primes divide it (with multiplicity for the "complete factor")
    # We need the small part = product of p^k where p < 24 and p^k || n
    # But for counting DISTINCT factors, we just need which primes appear
    
    # Actually, re-reading: "extract its complete factor composed of primes STRICTLY BELOW 24"
    # This means: for number m, small_part(m) = product of p^a where p in primes and p^a divides m
    # Then we collect all small_part values and count distinct ones
    
    # Wait - "MINIMIZE the number of distinct extracted factors" - this means distinct VALUES
    # of the small parts, not distinct primes!
    
    # Let me re-read: "extract its complete factor composed of primes STRICTLY BELOW 24. MINIMIZE the number of distinct extracted factors."
    # So for each n+i, we compute small_part(n+i) = the largest divisor of (n+i) that only has prime factors < 24
    # Then we look at the set {small_part(n+1), small_part(n+2), ..., small_part(n+24)}
    # and minimize |this set|
    
    # This is different from counting distinct primes! We're counting distinct VALUES of the small parts.
    
    # Precompute small_part for all numbers up to max_val
    small_part = [1] * (max_val + 1)
    for p in primes:
        p_power = p
        while p_power <= max_val:
            for m in range(p_power, max_val + 1, p_power):
                # Remove all factors of p from m, then multiply back p^a
                temp = m
                a = 0
                while temp % p == 0:
                    temp //= p
                    a += 1
                # small_part[m] should include p^a
                small_part[m] *= p ** a
            p_power *= p
    
    # Now for each n in [1000, 1000000], compute the number of distinct small_part values
    # in {small_part[n+1], ..., small_part[n+24]}
    
    # Use sliding window with frequency count
    best_n = 1000
    best_count = 25  # worst case is 24 distinct + 1
    
    # Initialize window for n=1000: numbers 1001 to 1024
    freq = {}
    distinct = 0
    for i in range(1, 25):
        sp = small_part[1000 + i]
        if sp not in freq:
            freq[sp] = 0
            distinct += 1
        freq[sp] += 1
    
    if distinct < best_count:
        best_count = distinct
        best_n = 1000
    
    # Slide the window
    for n in range(1001, max_n + 1):
        if time.time() > deadline:
            break
        
        # Remove n (which was n+0 in previous window, i.e., element that left)
        # Previous window: [n, n+1, ..., n+23] -> we had small_part[n+1]...small_part[n+24]
        # New window: [n+1, n+2, ..., n+24] -> small_part[n+2]...small_part[n+25]
        
        # Wait, let me re-index. For window starting at n:
        # Numbers are n+1, n+2, ..., n+24
        # So when we go from n to n+1:
        # Remove: small_part[n+1]
        # Add: small_part[n+25]
        
        remove_val = small_part[n + 1]
        freq[remove_val] -= 1
        if freq[remove_val] == 0:
            del freq[remove_val]
            distinct -= 1
        
        add_val = small_part[n + 25]
        if add_val not in freq:
            freq[add_val] = 0
            distinct += 1
        freq[add_val] += 1
        
        if distinct < best_count:
            best_count = distinct
            best_n = n
    
    return best_n
# EVOLVE-BLOCK-END
