"""Segmented sieve + Pollard-Brent verification for p-smooth runs."""
import math
import random
import time

# Deterministic Miller-Rabin bases
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime_mr(n):
    """Deterministic Miller-Rabin primality test."""
    if n < 2:
        return False
    if n == 2 or n == 3:
        return True
    if n % 2 == 0:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    r, d = 0, n - 1
    while d % 2 == 0:
        r += 1
        d //= 2
    for a in _MR_BASES:
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

def pollard_brent(n):
    """Pollard's Brent cycle detection factorization."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randint(1, n - 1)
        c = random.randint(1, n - 1)
        m = 128
        g, r, q = 1, 1, 1
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

def has_large_prime_factor(n, limit):
    """Check if n has any prime factor > limit. Returns True if it does."""
    if n <= 1:
        return False
    if n <= limit:
        return False
    if is_prime_mr(n):
        return n > limit
    # Try to find a factor
    d = pollard_brent(n)
    return has_large_prime_factor(d, limit) or has_large_prime_factor(n // d, limit)

def check_p_smooth(v, p):
    """Check if all prime factors of v are <= p."""
    if v <= 1:
        return True
    if v <= p:
        return True
    return not has_large_prime_factor(v, p)

def construct():
    """
    Find prime p and maximum k such that p^2+1, ..., p^2+k are all p-smooth.
    Uses segmented sieve over [p^2+1, p^2+200] for rapid screening.
    """
    best_k = 0
    best_p = 2
    
    # Precompute small primes for trial division (up to ~10^6)
    SIEVE_LIMIT = 10**6
    sieve = bytearray(b'\x01') * (SIEVE_LIMIT + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(SIEVE_LIMIT**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    small_primes = [i for i in range(2, SIEVE_LIMIT + 1) if sieve[i]]
    
    # For the segmented sieve approach:
    # For a given prime p, we want to check if p^2+i is p-smooth for i=1..200.
    # 
    # Key insight: p^2 + i has a prime factor > p iff there exists a prime q > p
    # such that q | (p^2 + i). Since q > p, gcd(q, p) = 1, so p^2 ≡ -i (mod q).
    # 
    # For the segmented sieve: for each prime q in (p, p+200], we can check which
    # i in [1, 200] satisfy p^2 ≡ -i (mod q), i.e., i ≡ -p^2 (mod q).
    # If i ≡ -p^2 (mod q) and 1 <= i <= 200, then q divides p^2 + i, making it non-p-smooth.
    #
    # Additionally, for primes q > p+200, if q divides p^2+i, then q >= p+201 > p,
    # and since p^2+i < p^2+201, we have q > p^2+i is impossible (q divides it).
    # So q must be <= p^2+i. But we also need q > p.
    # For large q, the constraint is tighter. However, for the segmented sieve,
    # we primarily care about primes q in (p, p^2+200].
    #
    # Actually, a simpler approach: for each i in 1..200, compute v = p^2+i.
    # v is p-smooth iff v has no prime factor > p.
    # 
    # Segmented sieve idea: For each prime q > p, mark all i in [1,200] such that
    # q | (p^2 + i). This means i ≡ -p^2 (mod q).
    # Since i is in [1, 200], we need i = (-p^2 mod q) if that's in [1,200],
    # or i = (-p^2 mod q) + q if that's in [1,200], etc.
    # But since q > p and i <= 200, and p^2 mod q = p^2 if p^2 < q, or p^2 - floor(p^2/q)*q otherwise.
    # 
    # For q > p^2 + 200, no i in [1,200] can satisfy q | (p^2+i) since p^2+i < q.
    # For p < q <= p^2+200, we can check i ≡ -p^2 (mod q).
    #
    # But iterating over all primes q in (p, p^2+200] is too expensive for large p.
    #
    # Better approach: For each i in 1..200, factor v = p^2+i using trial division
    # by small primes first, then use Pollard-Brent on the remainder if needed.
    # The segmented sieve helps by quickly identifying which i have small prime factors > p.
    
    start_time = time.time()
    time_limit = 110  # Leave 10s margin
    
    # Phase 1: Test small primes up to 10^4 with direct checking
    for p in range(2, 10000):
        if time.time() - start_time > time_limit:
            break
        if not is_prime_mr(p):
            continue
        k = 0
        for i in range(1, 201):
            v = p * p + i
            if not check_p_smooth(v, p):
                break
            k = i
        if k > best_k:
            best_k = k
            best_p = p
        if k == 200:
            break
    
    # Phase 2: Test primes up to 10^6 using segmented sieve optimization
    # For each prime p, use segmented sieve to quickly find which i in [1,200]
    # have a prime factor > p, then only apply Pollard-Brent to survivors.
    
    # Precompute primes up to 10^6 for candidate generation
    primes_up_to_1m = [i for i in range(2, SIEVE_LIMIT + 1) if sieve[i]]
    
    for p in primes_up_to_1m:
        if time.time() - start_time > time_limit:
            break
        if p < 10000:  # Already tested
            continue
        
        # Segmented sieve: mark i in [1,200] that have a prime factor > p
        # A number v = p^2 + i has a prime factor > p iff:
        # 1. v itself is prime and v > p, OR
        # 2. v has a composite factor with a prime component > p
        
        # Quick check: for each small prime q > p (but q <= 200 + something),
        # check if q divides any p^2 + i
        
        # Actually, let's use a different segmented sieve approach:
        # For each i in 1..200, we'll track the "remaining" part after dividing
        # by all primes <= p. If remaining > 1 and remaining > p, we need to check
        # if remaining has a prime factor > p.
        
        # More efficient: precompute for this p which i are "definitely bad"
        # by checking small primes q where p < q <= 200 (since if q > 200 and q | (p^2+i),
        # then q > p^2+i for small i, which is impossible unless q <= p^2+i)
        
        # Let's just do direct checking with early termination
        k = 0
        for i in range(1, 201):
            v = p * p + i
            # Quick rejection: if v is prime and v > p, it's not p-smooth
            # But checking primality of each v is expensive.
            # Instead, use trial division by primes up to min(sqrt(v), p)
            remaining = v
            smooth = True
            for sp in small_primes:
                if sp > p:
                    break
                if sp * sp > remaining:
                    break
                while remaining % sp == 0:
                    remaining //= sp
            if remaining == 1:
                smooth = True
            elif remaining <= p:
                smooth = True
            else:
                # remaining > p, need to check if it has a prime factor > p
                if is_prime_mr(remaining):
                    smooth = False
                else:
                    # Use Pollard-Brent to check
                    smooth = not has_large_prime_factor(remaining, p)
            
            if not smooth:
                break
            k = i
        
        if k > best_k:
            best_k = k
            best_p = p
        if k == 200:
            break
    
    # Phase 3: Test larger primes if time permits
    if time.time() - start_time < time_limit:
        # Sample primes in [10^6, 10^7]
        for p in range(1000001, 10000000, 2):
            if time.time() - start_time > time_limit:
                break
            if not is_prime_mr(p):
                continue
            k = 0
            for i in range(1, 201):
                v = p * p + i
                remaining = v
                smooth = True
                for sp in small_primes:
                    if sp > p:
                        break
                    if sp * sp > remaining:
                        break
                    while remaining % sp == 0:
                        remaining //= sp
                if remaining == 1:
                    smooth = True
                elif remaining <= p:
                    smooth = True
                else:
                    if is_prime_mr(remaining):
                        smooth = False
                    else:
                        smooth = not has_large_prime_factor(remaining, p)
                
                if not smooth:
                    break
                k = i
            
            if k > best_k:
                best_k = k
                best_p = p
            if k == 200:
                break
    
    return {'p': best_p, 'k': best_k}
