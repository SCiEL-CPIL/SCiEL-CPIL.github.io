"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

# EVOLVE-BLOCK-START

def _sigma(n):
    """Compute sigma(n) = sum of divisors of n."""
    if n <= 0:
        return 0
    result = 1
    temp = n
    for p in range(2, int(math.isqrt(n)) + 1):
        if temp % p == 0:
            factor_sum = 1
            power = 1
            while temp % p == 0:
                temp //= p
                power *= p
                factor_sum += power
            result *= factor_sum
    if temp > 1:
        result *= (1 + temp)
    return result

def _sigma_fast(n, primes):
    """Compute sigma(n) using precomputed primes list."""
    if n <= 1:
        return 1 if n == 1 else 0
    result = 1
    temp = n
    for p in primes:
        if p * p > temp:
            break
        if temp % p == 0:
            factor_sum = 1
            power = 1
            while temp % p == 0:
                temp //= p
                power *= p
                factor_sum += power
            result *= factor_sum
    if temp > 1:
        result *= (1 + temp)
    return result

def _sieve_primes(limit):
    """Generate primes up to limit using sieve."""
    is_prime = [True] * (limit + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_prime[i]:
            for j in range(i * i, limit + 1, i):
                is_prime[j] = False
    return [i for i in range(2, limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin before 120s limit
    
    # Strategy: enumerate k values that are products of small primes,
    # compute k*sigma(k), find M with most collisions.
    # Focus on k with many divisors since they have larger sigma(k)/k.
    
    # Generate primes up to a reasonable limit
    primes = _sieve_primes(1000)
    
    # Generate candidate k values: products of powers of small primes
    # k = 2^a * 3^b * 5^c * 7^d * ... where k <= some bound
    # We want k <= sqrt(M) and M <= 10^18, so k can be up to 10^9
    
    # Use a priority queue / BFS to generate numbers with many small prime factors
    # in increasing order, computing k*sigma(k) and tracking collisions
    
    # Approach: generate all numbers of the form 2^a * 3^b * 5^c * 7^d * 11^e * ...
    # up to a bound, compute their sigma, and look for collisions
    
    # For efficiency, generate numbers by prime factorization patterns
    # that are known to produce many divisors
    
    # Let's try a systematic approach: for each k up to a bound, compute k*sigma(k)
    # but focus on k with many small prime factors
    
    # Actually, let's do a direct enumeration but with early termination
    # and focus on finding the best M
    
    # Key insight: k*sigma(k) is multiplicative in a sense - if k = p^a, 
    # then k*sigma(k) = p^a * (1 + p + ... + p^a)
    
    # Let's enumerate k up to a reasonable bound and find collisions
    # Use a dictionary to track M -> list of k's
    
    # To make this faster, let's generate k values in a smart order
    # focusing on numbers with many divisors
    
    # Generate numbers of form 2^a * 3^b * 5^c * 7^d * 11^e * 13^f * 17^g
    # where the exponents are non-increasing (to avoid duplicates)
    
    max_k = 2_000_000  # We'll go up to this k value
    
    # For each k, compute k*sigma(k) and track collisions
    # But this is O(max_k) which might be too slow
    
    # Better: generate only numbers with many small prime factors
    # These are the ones most likely to have collisions
    
    # Let's use a different approach: generate k values by their prime factorization
    # and compute k*sigma(k) directly
    
    # Start with known good M values and try to find more k's for them
    
    # Known: M = 333312 has k = 336, 372, 434
    # Let's search for M values with more collisions
    
    # Strategy: enumerate k in ranges where k has many small prime factors
    # and compute k*sigma(k), then find the M with most k's
    
    # For speed, precompute sigma for small numbers using a sieve
    
    # Actually, let's just do a focused enumeration
    # Generate k = product of small primes with various exponents
    
    # Use a heap to generate numbers in increasing order
    import heapq
    
    # Generate numbers of form 2^a * 3^b * 5^c * 7^d * ... 
    # in increasing order using a heap
    # Each entry: (value, a, b, c, d, ...)
    
    # This is getting complex. Let's use a simpler approach:
    # Enumerate k up to a bound, but only for k that are "smooth" 
    # (all prime factors <= some threshold)
    
    # For k up to 50000, compute k*sigma(k) and find collisions
    # This should be fast enough
    
    bound = 50000
    M_to_ks = {}
    
    # Precompute smallest prime factor for fast factorization
    spf = list(range(bound + 1))
    for i in range(2, int(math.isqrt(bound)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, bound + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    for k in range(1, bound + 1):
        elapsed = time.time() - start_time
        if elapsed > deadline - 5:  # 5s safety margin
            break
        
        # Compute sigma(k) using spf
        temp = k
        sig = 1
        while temp > 1:
            p = spf[temp]
            power = 1
            factor_sum = 1
            while temp % p == 0:
                temp //= p
                power *= p
                factor_sum += power
            sig *= factor_sum
        
        M = k * sig
        if M <= 10**18:
            if M not in M_to_ks:
                M_to_ks[M] = []
            M_to_ks[M].append(k)
    
    # Find M with most k's
    best_M = None
    best_ks = []
    for M, ks in M_to_ks.items():
        if len(ks) > len(best_ks):
            best_M = M
            best_ks = ks
    
    if best_M is not None and len(best_ks) >= 3:
        # Verify all k's are valid
        valid_ks = []
        for k in best_ks:
            if k <= math.isqrt(best_M):
                # Verify k * sigma(k) == best_M
                temp = k
                sig = 1
                t = k
                while t > 1:
                    p = spf[t] if t <= bound else t
                    power = 1
                    factor_sum = 1
                    while t % p == 0:
                        t //= p
                        power *= p
                        factor_sum += power
                    sig *= factor_sum
                if k * sig == best_M:
                    valid_ks.append(k)
        
        if len(valid_ks) >= 3:
            return {'M': int(best_M), 'k': [int(k) for k in valid_ks[:1000]]}
    
    # Fallback to known solution
    return {'M': 333312, 'k': [336, 372, 434]}

# EVOLVE-BLOCK-END
