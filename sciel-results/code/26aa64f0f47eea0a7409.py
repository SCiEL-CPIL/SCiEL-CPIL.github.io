"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd

# EVOLVE-BLOCK-START
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin within 90s limit
    
    # Primes strictly below 24
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23]
    
    # We need to scan n in [1000, 1000000]
    # For each window [n+1, n+24], compute the 23-smooth part of each number
    # and count distinct values
    
    # Strategy: precompute 23-smooth parts for a large range, then slide window
    
    # The 23-smooth part of m is m with all prime factors >= 24 removed
    # i.e., divide m by all factors of primes >= 24
    
    # To be efficient, let's compute smooth_part[m] for m in a range
    # smooth_part[m] = m / (product of p^a for p >= 24 where p^a || m)
    
    # Actually, let's think differently. For each m, smooth_part[m] is the
    # largest divisor of m that is 23-smooth.
    
    # We can compute this by trial division by primes >= 24 up to sqrt(m),
    # but that's slow. Better: precompute using a sieve-like approach.
    
    # Alternative: for each m, smooth_part[m] = m. Then for each prime p >= 24,
    # for each multiple of p, divide out all powers of p.
    
    # But m goes up to 1000024, and primes >= 24 up to sqrt(1000024) ~ 1000.
    # That's manageable.
    
    max_m = 1000024  # max value of n+24 when n=1000000
    min_n = 1000
    
    # Precompute smooth parts
    smooth = list(range(max_m + 1))  # smooth[m] will hold the 23-smooth part of m
    
    # For each prime p >= 24, divide out all powers of p from multiples
    # Primes >= 24 up to max_m (but we only need to go up to max_m)
    # Actually, we need primes up to max_m, but for the division, we only
    # need to process primes up to max_m. However, for efficiency, we note
    # that if p > max_m, it can't divide any m in our range.
    # Primes >= 24 that can divide numbers up to max_m: 29, 31, 37, ...
    
    # Let's generate primes >= 24 up to max_m
    # But max_m is 1000024, generating all primes up to that is fine with a sieve
    
    # Sieve of Eratosthenes up to max_m
    is_prime = [True] * (max_m + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(max_m**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, max_m + 1, i):
                is_prime[j] = False
    
    # Now for each prime p >= 24, process multiples
    for p in range(24, max_m + 1):
        if is_prime[p]:
            # For each multiple of p, divide out all powers of p
            power = p
            while power <= max_m:
                for m in range(power, max_m + 1, power):
                    # Divide smooth[m] by p as long as divisible
                    while smooth[m] % p == 0:
                        smooth[m] //= p
                power *= p
    
    # Now smooth[m] is the 23-smooth part of m for m in [1, max_m]
    
    # Now scan windows: for n in [1000, 1000000], window is [n+1, n+24]
    # Count distinct values in smooth[n+1], smooth[n+2], ..., smooth[n+24]
    
    best_n = 1000
    best_count = float('inf')
    
    # Use a sliding window approach with a frequency counter
    # For the initial window at n=1000: numbers 1001 to 1024
    from collections import defaultdict
    
    freq = defaultdict(int)
    for i in range(1, 25):
        val = smooth[1000 + i]
        freq[val] += 1
    
    distinct = len(freq)
    if distinct < best_count:
        best_count = distinct
        best_n = 1000
    
    # Slide the window
    for n in range(1001, 1000001):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # Remove n (which is (n-1)+1) and add n+24
        old_val = smooth[n]  # this is (n-1)+1
        freq[old_val] -= 1
        if freq[old_val] == 0:
            del freq[old_val]
        
        new_val = smooth[n + 24]
        freq[new_val] += 1
        
        distinct = len(freq)
        if distinct < best_count:
            best_count = distinct
            best_n = n
            if best_count == 1:
                break  # Can't do better
    
    return best_n

# EVOLVE-BLOCK-END
