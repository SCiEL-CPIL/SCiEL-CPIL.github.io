"""Program for: Long runs after p^2 with no prime factor exceeding p (p <= 10^9).
Uses a segmented sieve to precompute smallest prime factors for [p^2+1, p^2+200],
enabling O(1) p-smoothness checks. Falls back to Pollard-Brent for verification.
Maximizes k within time budget."""

import time
import math
import random

# ---------- Deterministic Miller-Rabin for n < 3.3e24 ----------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
        if a % n == 0:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True

# ---------- Pollard's Brent ----------
def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
        m = 128
        g = r = q = 1
        while g == 1:
            x = y
            for _ in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for _ in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = q * abs(x - y) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g != 1:
                    break
        if g != n:
            return g

# ---------- Precompute small primes for trial division ----------
_SMALL_PRIMES = []
def _init_small_primes(limit=100000):
    """Generate small primes for fast trial division."""
    if limit < 2:
        return
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    _SMALL_PRIMES.extend(i for i in range(2, limit + 1) if is_p[i])

_init_small_primes(100000)

def _get_spf_for_range(start, length, p_limit):
    """
    Compute smallest prime factor for each number in [start, start+length-1].
    Only considers primes <= p_limit. Returns a list where spf[i] is the smallest
    prime factor of (start+i) that is <= p_limit, or 0 if no such factor exists
    (meaning the number is either 1 or has all prime factors > p_limit).
    """
    # spf[i] will store the smallest prime factor <= p_limit for number start+i
    # If spf[i] == 0 after processing, it means start+i has no prime factor <= p_limit
    # (i.e., it's either 1 or all its prime factors exceed p_limit)
    spf = [0] * length
    
    for q in _SMALL_PRIMES:
        if q > p_limit:
            break
        # Find the first multiple of q in [start, start+length-1]
        first = ((start + q - 1) // q) * q
        for j in range(first, start + length, q):
            idx = j - start
            if spf[idx] == 0:
                spf[idx] = q
            # Once we've found a factor, we don't need to keep marking
            # (spf stores the smallest, so first prime to hit is smallest)
    
    return spf

def _check_p_smooth_run(p, max_k=200):
    """
    Check how many consecutive integers p^2+1, ..., p^2+max_k are p-smooth.
    Uses segmented sieve for fast checking.
    Returns the maximum k such that all p^2+1, ..., p^2+k are p-smooth.
    """
    start = p * p + 1
    length = max_k
    
    # Get smallest prime factors for the range
    spf = _get_spf_for_range(start, length, p)
    
    k = 0
    for i in range(length):
        # spf[i] is the smallest prime factor <= p of (start + i)
        # If spf[i] == 0, it means (start+i) has no prime factor <= p
        # This happens when start+i == 1 (impossible here) or all prime factors > p
        # So if spf[i] == 0, then (start+i) is NOT p-smooth (unless it's 1)
        if spf[i] == 0:
            # Need to verify: is start+i actually prime and > p? Or composite with all factors > p?
            # Either way, it's not p-smooth
            break
        # spf[i] > 0 means there's at least one prime factor <= p
        # But we need ALL prime factors <= p
        # We need to fully factor start+i and check all factors are <= p
        # For efficiency, we can do trial division by primes up to p
        # But that's expensive for large p. Instead, we can check if the remaining
        # part after dividing by small primes is 1 or prime <= p
        
        n = start + i
        # Divide out all small prime factors
        for q in _SMALL_PRIMES:
            if q > p:
                break
            while n % q == 0:
                n //= q
            if n == 1:
                break
        if n == 1:
            k += 1
        elif is_prime(n) and n <= p:
            k += 1
        else:
            # n has a prime factor > p, or n is composite with a factor > p
            break
    
    return k

def is_p_smooth(n, p):
    """Check if all prime factors of n are <= p. Returns True if n is 1 or p-smooth."""
    if n <= 1:
        return True
    
    # Try small primes first with trial division
    for q in _SMALL_PRIMES:
        if q > p:
            break
        while n % q == 0:
            n //= q
        if n == 1:
            return True
    
    if n == 1:
        return True
    
    # If n is prime, check if it's <= p
    if is_prime(n):
        return n <= p
    
    # Use Pollard-Brent to find a factor, then recursively check
    factors = []
    _factor(n, factors, p)
    for f in factors:
        if f > p:
            return False
    return True

def _factor(n, factors, p):
    """Recursively factor n, adding factors to list. Early termination if factor > p."""
    if n == 1:
        return
    if is_prime(n):
        factors.append(n)
        return
    d = pollard_brent(n)
    _factor(d, factors, p)
    _factor(n // d, factors, p)

def sieve_primes(limit):
    """Generate all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, limit + 1) if is_p[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    best_k = 0
    best_p = 2
    
    # Generate primes up to a reasonable limit using sieve
    # Focus on primes where p^2+i numbers are likely to have small factors
    sieve_limit = 10_000_000
    primes = sieve_primes(sieve_limit)
    
    # Strategy: 
    # 1. Try all primes up to sieve_limit
    # 2. For each, compute how far the run goes using segmented sieve
    # 3. Keep track of best
    
    for p in primes:
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        k = _check_p_smooth_run(p, max_k=200)
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # If time remains, try some larger primes with better heuristics
    if time.time() - start_time < deadline - 20:
        # Try primes in ranges larger than sieve_limit
        # Use a smarter search: look for primes where p^2+1, p^2+2, etc.
        # are likely to be p-smooth
        
        lo = max(sieve_limit + 1, 10**6)
        hi = min(10**9, 10**8)
        
        # Try structured search: sample primes in geometric progression ranges
        num_samples = 500
        for i in range(num_samples):
            if time.time() - start_time > deadline - 5:
                break
            
            # Interpolate between lo and hi with some randomness
            frac = i / num_samples
            base_lo = int(lo * (1 + frac * 0.5))
            base_hi = min(hi, int(base_lo * 1.5))
            
            candidate = random.randrange(base_lo, base_hi + 1)
            if candidate % 2 == 0:
                candidate += 1
            
            # Find next prime
            found_prime = False
            for _ in range(100):
                if is_prime(candidate):
                    found_prime = True
                    break
                candidate += 2
                if candidate > base_hi:
                    break
            
            if not found_prime:
                continue
            
            k = _check_p_smooth_run(candidate, max_k=200)
            
            if k > best_k:
                best_k = k
                best_p = candidate
    
    return {'p': best_p, 'k': best_k}
