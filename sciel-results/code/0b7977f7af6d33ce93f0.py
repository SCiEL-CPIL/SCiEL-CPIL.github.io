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

def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        x = random.randrange(2, n)
        y = x
        c = random.randrange(1, n)
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d

def factor(n, factors=None):
    if factors is None:
        factors = []
    if n == 1:
        return factors
    if is_prime(n):
        factors.append(n)
        return factors
    d = pollard_rho(n)
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
    if is_prime(n):
        return n <= p
    
    # Trial divide by small primes up to min(p, sqrt(n))
    v = n
    sqrt_n = int(math.isqrt(n))
    limit = min(p, sqrt_n)
    
    for sp in small_primes:
        if sp > limit:
            break
        while v % sp == 0:
            if sp > p:
                return False
            v //= sp
    
    if v == 1:
        return True
    
    # Check if remaining v is prime
    if is_prime(v):
        return v <= p
    
    # v is composite and > p, or has factors > p
    # After dividing by all primes <= limit = min(p, sqrt_n):
    # - If v == 1: all factors were <= limit <= p, so p-smooth
    # - If v is prime: p-smooth iff v <= p
    # - If v is composite: v has a prime factor q. Since we divided by all primes <= limit,
    #   q > limit. If limit = p, then q > p, not p-smooth.
    #   If limit = sqrt_n < p, then q > sqrt_n. But v <= n, and if v = q * r with q <= r,
    #   then q <= sqrt(v) <= sqrt(n) = limit, contradiction. So v can't be composite here.
    # Therefore, if v > 1 and not prime, it's not p-smooth.
    
    return False

def construct():
    deadline = time.time() + 110  # safety margin before 120s kill
    
    best_k = 0
    best_p = 2
    
    # Precompute small primes for trial division - up to 100000
    small_primes = sieve_primes(100000)
    
    # Generate primes to search over
    # Start with primes up to 500000 - these are most likely to have long runs
    search_primes = sieve_primes(500000)
    
    for p in search_primes:
        if time.time() > deadline:
            break
        if p < 2:
            continue
        
        # Quick check: only check first 10 numbers to eliminate most primes
        base = p * p
        k_quick = 0
        for i in range(1, 11):
            n = base + i
            if is_p_smooth(n, p, small_primes):
                k_quick += 1
            else:
                break
        
        # If quick check shows at least 5 consecutive smooth numbers, do full check
        if k_quick >= 5:
            k = 0
            for i in range(1, 201):
                n = base + i
                if is_p_smooth(n, p, small_primes):
                    k += 1
                else:
                    break
            
            if k > best_k:
                best_k = k
                best_p = p
    
    # If we have time, try primes in larger ranges
    if best_k < 200 and time.time() < deadline - 30:
        # Try primes in ranges: 500k-2M, 2M-10M
        for start, end in [(500000, 2000000), (2000000, 10000000)]:
            if time.time() > deadline - 20:
                break
            # Generate primes in this range using segmented sieve
            # For simplicity, sample some primes
            # Use a window approach
            window_size = 100000
            pos = start
            while pos < end and time.time() < deadline - 20:
                sieve_end = min(pos + window_size, end)
                sieve = [True] * (sieve_end - pos)
                for sp in small_primes:
                    if sp * sp > sieve_end:
                        break
                    first = ((pos + sp - 1) // sp) * sp
                    for j in range(first - pos, sieve_end - pos, sp):
                        sieve[j] = False
                for i in range(sieve_end - pos):
                    if time.time() > deadline - 20:
                        break
                    cand = pos + i
                    if cand < 2:
                        continue
                    if sieve[i] and is_prime(cand):
                        # Quick check: only check first 10 numbers
                        base = cand * cand
                        k_quick = 0
                        for ii in range(1, 11):
                            n = base + ii
                            if is_p_smooth(n, cand, small_primes):
                                k_quick += 1
                            else:
                                break
                        
                        # If quick check shows at least 5 consecutive smooth numbers, do full check
                        if k_quick >= 5:
                            k = 0
                            for ii in range(1, 201):
                                n = base + ii
                                if is_p_smooth(n, cand, small_primes):
                                    k += 1
                                else:
                                    break
                            
                            if k > best_k:
                                best_k = k
                                best_p = cand
                pos = sieve_end
    
    if best_p == 0:
        best_p = 2
        best_k = 1
    
    return {'p': best_p, 'k': best_k}
