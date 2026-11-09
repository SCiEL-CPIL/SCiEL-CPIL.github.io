"""Program for: Long runs after p^2 with no prime factor exceeding p (p <= 10^9)."""

import time
import random
import math

# ---------- Deterministic Miller-Rabin for n < 3.3e24 ----------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def is_prime(n):
    if n < 2:
        return False
    for p in _MR_BASES:
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

def pollard_brent(n):
    """Brent's variant of Pollard's rho - faster than Floyd's."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
        m = random.randrange(1, 100)
        g = 1
        r = 1
        q = 1
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
                if g > 1:
                    break
        if g != n:
            return g

def factor(n, factors=None):
    if factors is None:
        factors = []
    if n == 1:
        return factors
    if is_prime(n):
        factors.append(n)
        return factors
    d = pollard_brent(n)
    factor(d, factors)
    factor(n // d, factors)
    return factors

def sieve_primes(limit):
    """Generate list of all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, limit + 1) if is_p[i]]

def is_p_smooth(n, p, small_primes):
    """Check if all prime factors of n are <= p.
    Uses trial division by small primes first, then Miller-Rabin/Pollard for remainder."""
    if n <= 1:
        return True
    
    # Quick check: if n is prime, it must be <= p
    if is_prime(n):
        return n <= p
    
    # Trial divide by small primes
    v = n
    sqrt_n = int(math.isqrt(n))
    limit = min(p, sqrt_n)
    
    for sp in small_primes:
        if sp > limit:
            break
        while v % sp == 0:
            v //= sp
    
    if v == 1:
        return True
    
    # Check if remaining v is prime
    if is_prime(v):
        return v <= p
    
    # v is composite. We need to check if all its factors are <= p.
    # Since we've divided out all primes <= min(p, sqrt_n),
    # if v is composite, it has a prime factor > min(p, sqrt_n).
    # If min(p, sqrt_n) = p, then that factor > p, so not p-smooth.
    # If min(p, sqrt_n) = sqrt_n < p, then v's smallest prime factor > sqrt_n,
    # but v <= n and if v = a*b with a <= b, then a <= sqrt(v) <= sqrt(n) = sqrt_n,
    # contradiction. So v can't be composite in this case.
    # Therefore, if v > 1 and not prime, it's not p-smooth.
    
    return False

def construct():
    deadline = time.time() + 110  # safety margin before 120s kill
    
    best_k = 0
    best_p = 2
    
    # Precompute small primes for trial division
    # Use primes up to 100000 for fast trial division
    small_primes = sieve_primes(100000)
    
    # Strategy: Search primes in increasing order, but skip many
    # Focus on primes that are more likely to have long runs
    # Small primes (p < 1000) often have longer runs because p^2 + i
    # has more small factors
    
    # Phase 1: Check all primes up to 10000 quickly
    search_primes_small = sieve_primes(10000)
    for p in search_primes_small:
        if time.time() > deadline:
            break
        
        k = 0
        base = p * p
        for i in range(1, 201):
            n = base + i
            if is_p_smooth(n, p, small_primes):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: Check primes from 10000 to 100000, sampling every few
    if best_k < 200 and time.time() < deadline - 50:
        primes_mid = sieve_primes(100000)
        # Sample every 10th prime to save time
        for p in primes_mid[1000::10]:  # skip first 1000 (already checked)
            if time.time() > deadline - 50:
                break
            
            k = 0
            base = p * p
            for i in range(1, 201):
                n = base + i
                if is_p_smooth(n, p, small_primes):
                    k += 1
                else:
                    break
            
            if k > best_k:
                best_k = k
                best_p = p
    
    # Phase 3: If still have time, check larger primes with segmented sieve
    if best_k < 200 and time.time() < deadline - 20:
        # Use segmented sieve for primes in larger ranges
        window_size = 50000
        for start in range(100000, 1000000, window_size):
            if time.time() > deadline - 20:
                break
            
            end = min(start + window_size, 1000000)
            sieve = [True] * (end - start)
            
            for sp in small_primes:
                if sp * sp > end:
                    break
                first = ((start + sp - 1) // sp) * sp
                for j in range(first - start, end - start, sp):
                    sieve[j] = False
            
            for i in range(end - start):
                if time.time() > deadline - 20:
                    break
                cand = start + i
                if cand < 2:
                    continue
                if sieve[i] and is_prime(cand):
                    k = 0
                    base = cand * cand
                    for ii in range(1, 201):
                        n = base + ii
                        if is_p_smooth(n, cand, small_primes):
                            k += 1
                        else:
                            break
                    
                    if k > best_k:
                        best_k = k
                        best_p = cand
    
    if best_p == 0:
        best_p = 2
        best_k = 1
    
    return {'p': best_p, 'k': best_k}
