"""Program for: Long runs after p^2 with no prime factor exceeding p (p <= 10^9)."""

import time
import math
import random

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
        m = random.randrange(1, n)
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
                if g > 1:
                    break
        if g != n:
            return g

def factor(n, factors=None):
    """Full factorization using trial division + Pollard-Brent."""
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
    """Generate all primes up to limit using sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def is_p_smooth(n, p, small_primes):
    """Check if all prime factors of n are <= p."""
    if n <= 1:
        return True
    if n <= p:
        return True  # n itself is <= p, so its only prime factor is <= p
    # Trial divide by small primes up to min(p, sqrt(n))
    rem = n
    for sp in small_primes:
        if sp > p:
            break
        if sp * sp > rem:
            break
        while rem % sp == 0:
            rem //= sp
    if rem == 1:
        return True
    if rem <= p:
        # rem is prime (since we divided by all primes up to sqrt(rem))
        return True
    # rem > p: check if rem is prime
    if is_prime(rem):
        return False  # rem > p and prime, so NOT p-smooth
    # rem is composite, factor it
    facs = factor(rem)
    return all(f <= p for f in facs)

def construct():
    deadline = time.time() + 145  # safety margin before 150s
    
    # Precompute small primes for trial division
    SMALL_PRIME_LIMIT = 1000000
    small_primes = sieve_primes(SMALL_PRIME_LIMIT)
    
    best_k = 0
    best_p = 0
    
    def check_prime(p):
        """Check all p^2+1 .. p^2+200 for p-smoothness, return max k."""
        nonlocal best_k, best_p
        k = 0
        p2 = p * p
        for i in range(1, 201):
            n = p2 + i
            if is_p_smooth(n, p, small_primes):
                k += 1
            else:
                break
            
            if time.time() > deadline - 2:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
        return k
    
    # Phase 1: Search small primes exhaustively
    primes_up_to_10k = sieve_primes(10000)
    for p in primes_up_to_10k:
        if time.time() > deadline - 5:
            break
        check_prime(p)
    
    # Phase 2: Search medium primes
    if time.time() < deadline - 10:
        primes_10k_to_100k = sieve_primes(100000)
        for p in primes_10k_to_100k:
            if p <= 10000:
                continue
            if time.time() > deadline - 5:
                break
            check_prime(p)
    
    # Phase 3: Search larger primes using segmented sieve for prime generation
    if time.time() < deadline - 15:
        ranges = [(100000, 1000000), (1000000, 10000000), (10000000, 100000000), (100000000, 1000000000)]
        
        for start, end in ranges:
            if time.time() > deadline - 10:
                break
            
            window_size = 1000000
            pos = start
            while pos < end and time.time() < deadline - 10:
                seg_end = min(pos + window_size, end)
                seg_size = seg_end - pos + 1
                seg_sieve = bytearray(b'\x01') * seg_size
                
                for sp in small_primes:
                    if sp * sp > seg_end:
                        break
                    first = ((pos + sp - 1) // sp) * sp
                    if first < sp * sp:
                        first = sp * sp
                    for j in range(first - pos, seg_size, sp):
                        seg_sieve[j] = 0
                
                for i in range(seg_size):
                    if time.time() > deadline - 10:
                        break
                    cand = pos + i
                    if cand < 2:
                        continue
                    if seg_sieve[i] and is_prime(cand):
                        check_prime(cand)
                
                pos = seg_end + 1
    
    if best_p == 0:
        best_p = 2
        best_k = 1
    
    return {'p': best_p, 'k': best_k}
