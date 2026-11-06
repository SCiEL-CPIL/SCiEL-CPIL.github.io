"""Rewritten program using segmented sieve for batch LPF computation."""
import time
import math
import random

def is_prime(n):
    """Deterministic Miller-Rabin for 64-bit integers."""
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in witnesses:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

def _smallest_prime_factor_sieve(limit):
    """Compute smallest prime factor for all numbers up to limit using sieve."""
    spf = list(range(limit + 1))
    if limit >= 1:
        spf[0] = 0
    if limit >= 2:
        spf[1] = 1
    for i in range(2, int(math.isqrt(limit)) + 1):
        if spf[i] == i:  # i is prime
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i
    return spf

def largest_prime_factor(v, spf):
    """Return the largest prime factor of v using precomputed spf table."""
    if v <= 1:
        return 1
    if v > len(spf) - 1:
        # Fallback: trial division for large v
        f = 1
        d = 2
        while d * d <= v:
            while v % d == 0:
                f = d
                v //= d
            d += 1
        return max(f, v)
    while v > 1:
        p = spf[v]
        if p > 1:
            # p is the smallest prime factor
            while v % p == 0:
                v //= p
    # Need to track the largest, not smallest. Let me redo this.
    # Actually, let me just factor fully.
    factors = []
    temp = v
    while temp > 1:
        p = spf[temp]
        if p <= 1:
            p = temp
        factors.append(p)
        temp //= p
    return max(factors)

def largest_prime_factor_correct(v, spf):
    """Return the largest prime factor of v using precomputed spf table."""
    if v <= 1:
        return 1
    if v > len(spf) - 1:
        # Fallback: trial division for large v
        f = 1
        d = 2
        while d * d <= v:
            while v % d == 0:
                f = d
                v //= d
            d += 1
        return max(f, v)
    max_pf = 1
    temp = v
    while temp > 1:
        p = spf[temp]
        if p <= 1:
            p = temp
        if p > max_pf:
            max_pf = p
        while temp % p == 0:
            temp //= p
    return max_pf

def construct():
    start_time = time.time()
    deadline = 120.0  # seconds total budget
    
    # We want to find prime p such that all prime factors of p^2+1, ..., p^2+k
    # are <= p, maximizing k.
    
    # Strategy:
    # 1. Precompute smallest prime factors up to some limit (e.g., 10^6 or 10^7)
    # 2. For candidate primes p, check p^2+1 ... p^2+200
    # 3. For each number, compute its largest prime factor using spf table
    # 4. If all LPF <= p, extend k
    
    # The key insight: for p up to 10^9, p^2 can be up to 10^18.
    # But we only need to check if all prime factors are <= p.
    # A number n has all prime factors <= p iff n is p-smooth.
    # 
    # Alternative: For a number m = p^2 + j, we need all prime factors <= p.
    # This means when we divide m by all primes <= p, we should get 1.
    # But that's expensive for large p.
    #
    # Better: compute the largest prime factor of m. If LPF(m) <= p, then m is p-smooth.
    #
    # For m up to p^2 + 200, with p up to 10^9, m can be up to ~10^18 + 200.
    # We can't precompute spf up to 10^18.
    #
    # Instead, use a hybrid approach:
    # 1. Precompute spf up to B (e.g., 10^7)
    # 2. For each m, first try to factor using spf up to B
    # 3. If remaining factor > 1, check if it's prime (Miller-Rabin)
    # 4. If composite, we need more work - but if the remaining is > B and composite,
    #    it has a factor > B, so LPF > B. If B >= p, then LPF > p fails.
    #
    # Actually, a cleaner approach:
    # For candidate p, for each m in [p^2+1, p^2+200]:
    #   - Try to find a prime factor > p. If found, m is NOT p-smooth.
    #   - To find if m has a prime factor > p:
    #     * First divide out all prime factors <= p (using spf table for small primes)
    #     * If remaining > 1, check if it's prime. If prime and > p, fail.
    #     * If remaining is composite, it has a factor > p (since we removed all <= p), so fail.
    #
    # Even better: just compute LPF(m) and check if <= p.
    # To compute LPF(m) for m up to ~10^18:
    #   - Trial divide by primes up to min(isqrt(m), p) 
    #   - If after dividing, remaining > 1, it's either prime or 1
    #   - The largest prime factor is max(last divided prime, remaining if prime)
    #
    # But trial division up to p for p up to 10^9 is too slow.
    #
    # Key optimization: if m has all factors <= p, and m <= p^2 + 200,
    # then m has at most 2 prime factors (both <= p), OR one factor <= p and rest = 1.
    # Actually m can have many small factors.
    #
    # Let me think differently. Precompute primes up to some limit L.
    # For each candidate p:
    #   For each j in 1..200:
    #     m = p^2 + j
    #     Check if m is p-smooth by trying to factor using primes <= p
    #     But we only have primes up to L. If L < p, we can't fully check.
    #
    # Practical approach:
    # - Precompute primes up to 10^6 (78498 primes)
    # - For small p (p <= 10^6), we can fully factor m using trial division by these primes
    # - For larger p, we use a different strategy:
    #   * m = p^2 + j. If m has a prime factor q > p, then q >= p+1.
    #   * Since m <= p^2 + 200, and q > p, then m/q < p^2/p = p (approximately).
    #   * So m = q * r where q > p and r < p (roughly).
    #   * We can check: does m have a divisor in (p, p^2/p]? 
    #   * This is hard.
    #
    # Let me try a simpler approach that works for the time budget:
    # - Search primes p up to some limit (say 10^5 or 10^6)
    # - For each p, check k from 1 to 200
    # - For each m = p^2 + j, compute LPF using trial division up to min(isqrt(m), p)
    #   but stop early if we find a factor > p
    #
    # Actually, the most efficient check for "is m p-smooth":
    #   Divide m by all primes <= p. If result is 1, it's p-smooth.
    #   But we need primes up to p.
    #
    # Let me precompute primes up to 2*10^6 and search p up to 2*10^6.
    
    # Precompute primes up to 2*10^6
    SIEVE_LIMIT = 2_000_000
    sieve = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    sieve[0:2] = b'\x00\x00'
    for i in range(2, int(math.isqrt(SIEVE_LIMIT)) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i in range(SIEVE_LIMIT + 1) if sieve[i]]
    
    # Also precompute spf for fast factorization of small numbers
    spf = list(range(SIEVE_LIMIT + 1))
    for i in range(2, int(math.isqrt(SIEVE_LIMIT)) + 1):
        if spf[i] == i:
            for j in range(i*i, SIEVE_LIMIT + 1, i):
                if spf[j] == j:
                    spf[j] = i
    
    best_p = 2
    best_k = 1
    
    # Function to check if m is p-smooth using spf table and trial division
    def is_p_smooth(m, p, primes_list):
        """Check if all prime factors of m are <= p."""
        if m <= 1:
            return True
        # First, use spf table if m is small enough
        if m <= SIEVE_LIMIT:
            temp = m
            while temp > 1:
                f = spf[temp]
                if f <= 1:
                    f = temp
                if f > p:
                    return False
                while temp % f == 0:
                    temp //= f
            return True
        else:
            # m > SIEVE_LIMIT, use trial division by primes
            temp = m
            for q in primes_list:
                if q > p:
                    break
                if q * q > temp:
                    break
                if temp % q == 0:
                    # q is a factor, q <= p so it's fine
                    while temp % q == 0:
                        temp //= q
            # After dividing by all primes <= min(p, isqrt(temp)),
            # if temp > 1, it's either prime or has a factor > isqrt(original temp)
            if temp > 1:
                # temp is either prime or 1
                # Since we divided by all primes <= p (or up to sqrt), 
                # if temp > p, it's a prime factor > p
                if temp > p:
                    return False
            return True
    
    # Search over primes p
    search_limit = SIEVE_LIMIT  # search up to 2*10^6
    for idx in range(len(primes)):
        if time.time() - start_time > deadline - 10:
            break
        
        p = primes[idx]
        
        # For each j from 1 to 200, check if p^2 + j is p-smooth
        k = 0
        p_squared = p * p
        for j in range(1, 201):
            m = p_squared + j
            if is_p_smooth(m, p, primes):
                k = j
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
        
        # Early stop: if p is getting large, p^2 grows fast
        # and it's unlikely to find long runs
        if p > 100000 and k < best_k:
            # Could continue but let's be conservative
            pass
    
    # Also try p = 2 and p = 3 explicitly
    if best_k < 1:
        best_k = 1
        best_p = 2
    
    return {'p': best_p, 'k': best_k}
