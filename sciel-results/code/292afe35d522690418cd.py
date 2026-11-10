"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math

# EVOLVE-BLOCK-START

def _sieve_primes(limit):
    """Return list of primes up to limit."""
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    # Precompute primes up to sqrt(1000000) = 1000 for initial filtering
    # and up to a larger bound for the segmented sieve
    primes_small = _sieve_primes(1000)
    primes_large = _sieve_primes(100000)
    
    # Segmented sieve to compute largest prime factor for range [10000, 1000000]
    # We'll compute lpf for each number in the range
    # Range size: 990001 numbers
    # For each number, we want its largest prime factor
    
    # Strategy: 
    # 1. Initialize an array lpf[0..N] where N = 1000000
    # 2. For each prime p, mark multiples of p in the range, updating lpf to max(current, p)
    # 3. For remaining numbers > 1 that are not marked (i.e., they are prime themselves), lpf = the number
    # 4. Then find n in [10000, 999999] that minimizes max(lpf[n], lpf[n+1])
    
    N = 1000000
    # lpf[i] = largest prime factor of i, 0 if not yet determined
    lpf = [0] * (N + 1)
    
    # For each prime p, mark multiples in [10000, 1000000]
    # We need primes up to 1000000 to mark all composites
    # But we can use the fact that if a number x in [10000, 1000000] has no prime factor <= sqrt(x), then x is prime
    
    # Actually, let's use a more efficient approach:
    # For each prime p, for each multiple m of p in [10000, 1000000], set lpf[m] = max(lpf[m], p)
    # This is O(N log log N) if done carefully
    
    # But N = 1000000, and we have ~9592 primes up to 100000
    # For each prime p, we mark N/p multiples
    # Total work: sum_{p<=100000} N/p ≈ N * log(log(100000)) ≈ 1000000 * 3.5 = 3.5 million operations
    # This is fast
    
    # Let's do it
    for p in primes_large:
        if p > N:
            break
        # Mark multiples of p in [10000, 1000000]
        start = max(10000, p)  # smallest multiple of p >= 10000
        # Actually, we need the smallest multiple of p >= 10000
        if start % p != 0:
            start += p - (start % p)
        # But we also need start >= p, which is satisfied since p >= 2 and start >= 10000
        for m in range(start, N + 1, p):
            if lpf[m] < p:
                lpf[m] = p
    
    # Now, for numbers in [10000, 1000000] that are still 0, they are prime
    # (since if they were composite, they'd have a prime factor <= sqrt(x) <= 1000, 
    #  and we marked all multiples of primes up to 100000)
    # Wait, that's not quite right. A composite number x has a prime factor <= sqrt(x).
    # If x is in [10000, 1000000], sqrt(x) <= 1000.
    # So if x is composite, it has a prime factor <= 1000.
    # We marked multiples of all primes up to 100000, so all composites are marked.
    # So lpf[m] == 0 means m is prime.
    
    for m in range(10000, N + 1):
        if lpf[m] == 0:
            lpf[m] = m
    
    # Now find the best n in [10000, 999999] that minimizes max(lpf[n], lpf[n+1])
    best_val = float('inf')
    best_n = 10000
    
    for n in range(10000, 1000000):
        if time.time() > deadline:
            break
        
        candidate = max(lpf[n], lpf[n + 1])
        if candidate < best_val:
            best_val = candidate
            best_n = n
    
    return best_n

# EVOLVE-BLOCK-END
