import time
import math
import random

# ── Deterministic Miller-Rabin for n < 2^64 ──────────────────────────────────
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

_SMALL_PRIMES_MR = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37,
                    41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97,
                    101, 103, 107, 109, 113, 127, 131, 137, 139, 149,
                    151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]

def is_prime(n):
    """Deterministic Miller-Rabin. Exact for n < 2^64 with 12 base primes."""
    if n < 2:
        return False
    for p in _SMALL_PRIMES_MR:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

def _gen_primes_upto(limit):
    """Sieve of Eratosthenes to generate primes up to limit."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if sieve[i]:
            start = i * i
            step = i
            sieve[start::step] = bytearray(len(sieve[start::step]))
    return [i for i, is_p in enumerate(sieve) if is_p]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_k = 0
    best_p = 2
    
    # Generate small primes for stripping
    # We need primes up to a reasonable limit to speed up the check.
    # If we strip up to 10^6, we handle many factors.
    SMALL_PRIME_LIMIT = 100_000
    small_primes = _gen_primes_upto(SMALL_PRIME_LIMIT)
    
    def is_p_smooth_fast(n, p):
        """
        Check if n is p-smooth.
        Logic:
        1. Strip small prime factors.
        2. If remainder r == 1, smooth.
        3. If r > 1:
           - If r is prime, it is smooth iff r <= p.
           - If r is composite, it is smooth (since sqrt(r) < p+1, so all factors <= p).
        """
        r = n
        for q in small_primes:
            if q * q > r:
                break
            if r % q == 0:
                while r % q == 0:
                    r //= q
                if r == 1:
                    return True
        if r == 1:
            return True
        
        # Check if r is prime
        if is_prime(r):
            return r <= p
        else:
            # r is composite, so it has a factor <= sqrt(r) < p+1, so <= p
            return True

    # Phase 1: Sieve and test primes up to 10^7
    PRIME_LIMIT = 10_000_000
    primes = _gen_primes_upto(PRIME_LIMIT)
    
    for p in primes:
        if time.time() > deadline:
            break
        base = p * p
        k = 0
        for i in range(1, 201):
            val = base + i
            if not is_p_smooth_fast(val, p):
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 2: Scan larger primes
    # We can scan from 10^7 upwards.
    # Use Miller-Rabin to find primes.
    # To be efficient, we can skip multiples of small primes.
    # But simple odd scanning with Miller-Rabin is okay for a while.
    
    current = PRIME_LIMIT + 1
    if current % 2 == 0:
        current += 1
        
    # Scan up to 50M or until time runs out
    SCAN_LIMIT = 50_000_000
    while current <= SCAN_LIMIT and time.time() < deadline - 10:
        if is_prime(current):
            p = current
            base = p * p
            k = 0
            for i in range(1, 201):
                val = base + i
                if not is_p_smooth_fast(val, p):
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p
        current += 2
        
    # Phase 3: Random sampling for larger primes
    # Larger p might have longer runs?
    # The probability of p-smoothness is roughly constant for u=2.
    # But the "composite remainder is smooth" rule helps.
    # If p is large, the remainder r is more likely to be composite?
    # Actually, r is n divided by small factors.
    # If n has no small factors, r=n.
    # n is prime with prob 1/ln(n) ~ 1/(2 ln p).
    # So n is composite with high prob.
    # If n is composite, it is p-smooth.
    # So for large p, p^2+i is p-smooth with high probability!
    # Wait.
    # If n is composite, is it always p-smooth?
    # Yes, if sqrt(n) < p+1.
    # n = p^2 + i. sqrt(n) < p+1.
    # So ANY composite n in the range is p-smooth.
    # ONLY prime n in the range can be NOT p-smooth (if n > p, which is true).
    # So p^2 + i is NOT p-smooth iff p^2 + i is PRIME.
    # This simplifies everything!
    
    # Let's verify.
    # If n is composite, let q be smallest prime factor. q <= sqrt(n) < p+1. So q <= p.
    # So all factors are <= p.
    # So n is p-smooth.
    # If n is prime, n > p (since n = p^2 + i > p).
    # So n has a factor n > p.
    # So n is NOT p-smooth.
    
    # So the condition "p^2 + i is p-smooth" is equivalent to "p^2 + i is COMPOSITE".
    # We just need to find a prime p such that p^2 + 1, ..., p^2 + k are all COMPOSITE.
    
    # This changes the algorithm drastically.
    # We just need to check if p^2 + i is prime.
    # If it is prime, the run stops.
    # If it is composite, we continue.
    
    # This is much faster!
    # We don't need smoothness checks, just primality tests.
    
    # Let's restart the logic with this insight.
    
    best_k = 0
    best_p = 2
    
    def is_composite(n):
        return not is_prime(n)

    # Re-scan with the new, faster logic
    # We can scan many more primes now.
    
    # Reset time check
    if time.time() > deadline - 5:
        return {'p': best_p, 'k': best_k}
        
    # Re-use the primes list from Phase 1
    # But we need to re-evaluate them with the faster check?
    # The old check was correct, just slow.
    # The new check is "is_prime(p^2+i)".
    # Let's re-run the search with the new check.
    
    # We can reuse the primes list.
    for p in primes:
        if time.time() > deadline:
            break
        base = p * p
        k = 0
        for i in range(1, 201):
            val = base + i
            if is_prime(val):
                break
            k += 1
        if k > best_k:
            best_k = k
            best_p = p
            
    # Continue scanning larger primes
    current = PRIME_LIMIT + 1
    if current % 2 == 0:
        current += 1
        
    SCAN_LIMIT = 100_000_000
    while current <= SCAN_LIMIT and time.time() < deadline - 5:
        if is_prime(current):
            p = current
            base = p * p
            k = 0
            for i in range(1, 201):
                val = base + i
                if is_prime(val):
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p
        current += 2
        
    # Random sampling for very large primes
    while time.time() < deadline - 2:
        p_candidate = random.randint(100_000_000, 1_000_000_000)
        if p_candidate % 2 == 0:
            p_candidate += 1
            
        # Quick filter
        if is_prime(p_candidate):
            p = p_candidate
            base = p * p
            k = 0
            for i in range(1, 201):
                val = base + i
                if is_prime(val):
                    break
                k += 1
            if k > best_k:
                best_k = k
                best_p = p

    return {'p': best_p, 'k': best_k}
