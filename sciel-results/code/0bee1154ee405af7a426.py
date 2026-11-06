import time
import math
import random

# Deterministic Miller-Rabin for 64-bit integers
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return n == 2
    if n % 3 == 0:
        return n == 3
    small_primes = (5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for sp in small_primes:
        if n % sp == 0:
            return n == sp
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
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True

_SMALL_TRIAL = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251)

def _is_p_smooth(n, p):
    """Check if all prime factors of n are <= p. Returns True if so."""
    if n <= 1:
        return True
    if n <= p:
        return True
    for sp in _SMALL_TRIAL:
        if sp > p:
            break
        if n % sp == 0:
            while n % sp == 0:
                n //= sp
            if n == 1:
                return True
            if n <= p:
                return True
            if is_prime(n):
                return n <= p
    # Now n > p and not divisible by small primes up to min(p, 251)
    # Check if n itself is prime
    if is_prime(n):
        return n <= p
    # n is composite and > p, need to find a factor
    # If n has a factor <= p, we need to find it
    # Try a few more trial divisions if p is not too large
    # For large p, n could have a large factor > p
    # We need to factor n enough to determine if all factors are <= p
    # Use a simple approach: try to find any factor
    # If we find a factor f, check if f <= p and n/f is p-smooth
    # But we don't want full factorization for speed
    
    # Try trial division up to a small limit (since p can be large, but we've done small primes)
    # Actually, the key insight: if n is composite and all its prime factors are <= p,
    # then n <= p^2 (since n = p^2 + i and i <= 200, so n is just above p^2)
    # Wait, n = p^2 + i, so n > p^2. If all prime factors of n are <= p,
    # then n can be written as product of primes each <= p.
    # Since n > p^2, we need at least one factor > 1, and actually the product
    # of primes <= p can be > p^2 only if we have multiple factors.
    # For example, p=100, p^2=10000, n=10001=73*137, but 137>100 so not p-smooth.
    # n=10001, p=100: 10001/73=137>100, so not p-smooth.
    # But n=10000+1=10001. What about n where all factors are <= p?
    # e.g., p=5, p^2=25, n=26=2*13, 13>5, not p-smooth.
    # p=7, p^2=49, n=50=2*5*5, all <=7, so p-smooth! k>=1.
    # n=51=3*17, 17>7, not p-smooth.
    
    # The key: for n = p^2 + i to be p-smooth with i small,
    # n must factor into primes all <= p.
    # Since n > p^2, if n is composite, it has a factor >= 2.
    # The smallest possible prime factor is 2.
    # If n has a prime factor q > p, then n is NOT p-smooth.
    # So we need to check: does n have any prime factor > p?
    # Equivalent: can n be factored completely into primes <= p?
    
    # For efficiency, let's try to find a factor of n.
    # If we find a factor f of n:
    #   - If f > p and is_prime(f), then not p-smooth.
    #   - If f > p, need to check if f itself has all prime factors <= p.
    #   - If f <= p, check n/f is p-smooth.
    
    # Simple approach: try a limited number of trial divisions
    # and if that fails, use a gcd-based or simple factor finding
    
    # Let's try: if n < p*p*2, then any factor f of n satisfies f <= sqrt(n) < p*sqrt(2) ~ 1.414*p
    # So if we find a factor f > p, and f is prime, we're done (not p-smooth).
    # If f is composite, we need to recurse.
    
    # For now, let's do a bounded trial division and Miller-Rabin check
    # If n passes small prime trial division and is not prime,
    # and n > p, we do a simple factor search
    
    # Actually, let's just check: is n prime? (already did)
    # n is composite. Try to find a factor.
    # Use trial division up to a reasonable limit
    limit = min(10000, int(math.isqrt(n)) + 1)
    d = 2
    while d <= limit and d <= p:
        if n % d == 0:
            f = d
            rest = n // f
            return _is_p_smooth(f, p) and _is_p_smooth(rest, p)
        d += 1 if d == 2 else 2  # 2, 3, 5, 7, ...
    
    # If we didn't find a small factor, n might have all factors > limit
    # If n > p and is composite, and all prime factors > limit,
    # then the smallest prime factor > limit.
    # If limit >= p, then we've checked all possible factors <= p,
    # so if no factor found, all prime factors > p, hence not p-smooth.
    # But we only checked up to min(10000, sqrt(n), p).
    
    # If p <= 10000, we've checked all primes <= p, so n is not p-smooth.
    if p <= 10000:
        return False
    
    # If p > 10000 and no factor found up to 10000,
    # n could still be p-smooth with all factors > 10000 but <= p.
    # This is unlikely for n = p^2 + i with small i, but possible.
    # In this case, we do a Pollard-Rho to find a factor.
    d = _pollard_rho(n)
    if d is None or d == n:
        # Shouldn't happen for composite n, but safety
        return is_prime(n) and n <= p
    return _is_p_smooth(d, p) and _is_p_smooth(n // d, p)

def _pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    while True:
        c = random.randrange(1, n)
        x = random.randrange(2, n - 1)
        y = x
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d

def _sieve(limit):
    """Return list of primes up to limit using bytearray sieve."""
    if limit < 2:
        return []
    flags = bytearray([1]) * (limit + 1)
    flags[0] = flags[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if flags[i]:
            flags[i * i: limit + 1: i] = bytearray(len(range(i * i, limit + 1, i)))
    return [i for i in range(2, limit + 1) if flags[i]]

def _run_length(p, max_k=200):
    """Compute the run length starting at p^2+1."""
    p_sq = p * p
    k = 0
    for i in range(1, max_k + 1):
        n = p_sq + i
        if not _is_p_smooth(n, p):
            break
        k += 1
    return k

def construct():
    t0 = time.time()
    deadline = t0 + 145  # 5s safety margin under 150s limit

    best_k = 0
    best_p = 2

    # Phase 1: Sieve and check all primes up to 50 million
    limit1 = 50_000_000
    primes = _sieve(limit1)

    for p in primes:
        if time.time() > deadline:
            break
        k = _run_length(p)
        if k > best_k:
            best_k = k
            best_p = p
        if k >= 200:
            break

    # Phase 2: Check primes in [50M, 200M] using a segmented sieve approach
    # For larger p, the run length is typically very short (1-3),
    # but we still check to be thorough.
    if time.time() < deadline - 5:
        # Use trial division with small primes for primality testing
        small_primes_for_test = _sieve(100_000)
        
        lo = limit1 + 1
        hi = 200_000_000
        chunk_size = 500_000
        
        for start in range(lo, hi, chunk_size):
            if time.time() > deadline:
                break
            end = min(start + chunk_size, hi)
            
            # Check odd numbers only
            for n in range(start if start % 2 == 1 else start + 1, end + 1, 2):
                if time.time() > deadline:
                    break
                # Primality test via trial division
                is_p = True
                for sp in small_primes_for_test:
                    if sp * sp > n:
                        break
                    if n % sp == 0:
                        if n == sp:
                            is_p = True
                        else:
                            is_p = False
                        break
                if is_p:
                    k = _run_length(n)
                    if k > best_k:
                        best_k = k
                        best_p = n
                    if k >= 200:
                        return {'p': best_p, 'k': best_k}

    return {'p': best_p, 'k': best_k}
