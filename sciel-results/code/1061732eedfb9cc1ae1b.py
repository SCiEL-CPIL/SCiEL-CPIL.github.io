"""Rewritten program: segmented-sieve factoring over [p^2+1, p^2+200] for many candidate primes."""
import time
import math

DEADLINE = 130.0  # safety margin below 150s wall-clock

def _is_prime_small(n, primes_list):
    """Check primality using precomputed primes up to sqrt(n)."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    limit = int(math.isqrt(n))
    for p in primes_list:
        if p > limit:
            break
        if n % p == 0:
            return False
    return True

def _sieve_primes_up_to(n):
    """Generate all primes up to n using simple sieve."""
    if n < 2:
        return []
    is_p = bytearray(b'\x01') * (n + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(math.isqrt(n)) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, n + 1) if is_p[i]]

def _factor_p_smooth_run(p, primes_list, max_k=200):
    """
    For prime p, find the maximum k (1..max_k) such that every integer
    p^2+1, ..., p^2+k has all prime factors <= p.
    
    Uses segmented sieve: for each prime q <= p, mark multiples in [p^2+1, p^2+max_k].
    An integer n in the range is p-smooth iff after removing all prime factors q<=p,
    the remainder is 1.
    """
    base = p * p
    # We'll track the "remaining" part of each number after dividing out all primes <= p
    # rem[i] corresponds to base + 1 + i
    rem = [base + 1 + i for i in range(max_k)]
    
    for q in primes_list:
        if q > p:
            break
        # Find the first multiple of q in [base+1, base+max_k]
        start = ((base + 1 + q - 1) // q) * q
        # Actually: first m such that base+1 <= m and m % q == 0
        # m = ceil((base+1)/q) * q
        start = ((base + 1 + q - 1) // q) * q
        if start < base + 1:
            start += q
        for m in range(start, base + max_k + 1, q):
            idx = m - base - 1
            if idx < 0 or idx >= max_k:
                continue
            while rem[idx] % q == 0:
                rem[idx] //= q
    
    # Now rem[i] should be 1 if base+1+i is p-smooth
    k = 0
    for i in range(max_k):
        if rem[i] == 1:
            k = i + 1
        else:
            break
    return k

def construct():
    t_start = time.time()
    
    # Precompute primes up to a reasonable limit for primality testing and factoring.
    # We need primes up to p for factoring, but p can be up to 10^9.
    # Strategy: precompute primes up to 10^6 (enough for quick primality tests of p up to 10^12),
    # and for factoring we iterate over these primes. For p > 10^6, we only need primes up to p,
    # but the interval [p^2+1, p^2+200] is small so we only need to divide by primes q where
    # q <= p AND q divides some number in the interval. Since the interval is only 200 wide,
    # any prime q > 200 can divide at most one number in the interval. So we can handle large q separately.
    
    # Actually, let's think more carefully. For a given p, we need to check if each n in [p^2+1, p^2+k]
    # has all prime factors <= p. The key insight: if q is a prime with q > p, then q cannot divide n
    # unless n >= q. But n is around p^2, so q could be as large as p^2. However, we only need to
    # verify that no prime factor exceeds p.
    
    # Better approach: For each n in [p^2+1, p^2+200], factor it completely and check the largest
    # prime factor. But that's expensive.
    
    # Segmented approach: For primes q <= p, divide out all factors of q from each n.
    # After processing all q <= p, if rem[n] == 1, then n is p-smooth.
    # The issue: we need all primes up to p. For p up to 10^7, that's ~664k primes.
    # For p up to 10^9, that's ~50M primes - too many to store.
    
    # Alternative: Only primes q <= p that could possibly divide any n in [p^2+1, p^2+200] matter.
    # If q > 200, then q divides at most one n in the interval. For such q, we can check directly.
    # If q <= 200, there are only 46 such primes.
    
    # Revised strategy:
    # 1. Precompute small primes up to 200 (or a bit more).
    # 2. For each candidate prime p:
    #    a. For each small prime q <= min(p, 200): divide out q from all n in interval.
    #    b. For remaining n > 1, check if n is prime or composite with all factors > p.
    #       Actually, after removing small factors, the remaining part r of n either:
    #       - is 1 (good)
    #       - is a prime q > p (bad)
    #       - is composite with all prime factors > p (bad, since at least one factor > p)
    #       - has a prime factor <= p that we missed (shouldn't happen if we removed all q <= p)
    
    # Wait, this doesn't work because we need to remove ALL prime factors <= p, not just small ones.
    # For p = 10^7, there are ~664k primes <= p. We can't iterate over all of them for each candidate.
    
    # Let me reconsider. The interval [p^2+1, p^2+200] has 200 numbers. For a prime q to divide
    # any number in this interval, we need q <= p^2+200. But we only care about q <= p.
    # For q <= p, q can divide at most ceil(200/q) numbers in the interval.
    # Total work per candidate p: sum over primes q <= p of ceil(200/q) ≈ 200 * sum(1/q) for q<=p ≈ 200 * log(log(p))
    # For p = 10^7: log(log(10^7)) ≈ log(16.1) ≈ 2.78, so ~556 operations per candidate.
    # Number of primes up to 10^7: ~664k. Total: 664k * 556 ≈ 370M operations. Too slow in Python.
    
    # Need to be smarter. Let's limit p to a smaller range first and see what k we can achieve.
    
    # Precompute primes up to 10^6 for primality testing
    SIEVE_LIMIT = 10**6
    primes_small = _sieve_primes_up_to(SIEVE_LIMIT)
    
    # Also precompute primes up to 200 for the small-prime division step
    primes_200 = [q for q in primes_small if q <= 200]
    
    best_k = 0
    best_p = 2
    
    # Try p from small primes upward
    # For each prime p, compute the p-smooth run length
    
    # To be efficient, we'll only try primes p where p is not too large initially,
    # then expand if time permits.
    
    # Let's try primes up to 10^5 first
    max_p_search = 10**5
    primes_to_try = [q for q in primes_small if q <= max_p_search]
    
    for p in primes_to_try:
        if time.time() - t_start > DEADLINE - 10:
            break
        
        base = p * p
        # rem[i] = base + 1 + i, we'll divide out prime factors
        rem = [base + 1 + i for i in range(200)]
        
        # Divide out all primes q <= p
        # Optimization: only primes q where q <= p AND q <= base+200 (always true for q <= p < base)
        # For efficiency, iterate over primes_small up to p
        for q in primes_small:
            if q > p:
                break
            # Find first multiple of q in [base+1, base+200]
            # m = ceil((base+1)/q) * q
            start = ((base + 1 + q - 1) // q) * q
            for m in range(start, base + 201, q):
                idx = m - base - 1
                if 0 <= idx < 200:
                    while rem[idx] % q == 0:
                        rem[idx] //= q
        
        # Count consecutive 1s from the start
        k = 0
        for i in range(200):
            if rem[i] == 1:
                k = i + 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            # print(f"p={p}, k={k}")
    
    # If we have time, try larger primes
    if time.time() - t_start < DEADLINE - 20:
        # Try primes from 10^5 to 10^6
        primes_to_try2 = [q for q in primes_small if 10**5 < q <= 10**6]
        for p in primes_to_try2:
            if time.time() - t_start > DEADLINE - 10:
                break
            
            base = p * p
            rem = [base + 1 + i for i in range(200)]
            
            for q in primes_small:
                if q > p:
                    break
                start = ((base + 1 + q - 1) // q) * q
                for m in range(start, base + 201, q):
                    idx = m - base - 1
                    if 0 <= idx < 200:
                        while rem[idx] % q == 0:
                            rem[idx] //= q
            
            k = 0
            for i in range(200):
                if rem[i] == 1:
                    k = i + 1
                else:
                    break
            
            if k > best_k:
                best_k = k
                best_p = p
    
    # Ensure k >= 1
    if best_k < 1:
        best_k = 1
        best_p = 2
    
    return {'p': best_p, 'k': best_k}
