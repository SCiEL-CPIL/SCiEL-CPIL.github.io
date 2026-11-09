"""Program for: Long runs after p^2 with no prime factor exceeding p (p <= 10^9)."""

import time
import random
import math

# ---------- Deterministic Miller-Rabin for n < 3.3e24 ----------
_MR_BASES_64 = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def is_prime(n):
    if n < 2:
        return False
    for p in _MR_BASES_64:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES_64:
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
    """Pollard's rho with Brent's improvement."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
        m = random.randrange(1, n)
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
                if g != 1:
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

def check_run_smooth(base, p, k, small_primes):
    """Check if base+1 through base+k are all p-smooth.
    Returns the length of the longest consecutive run starting at base+1.
    Uses batched trial division for speed."""
    if k <= 0:
        return 0
    
    # We'll track remainders for each number
    # nums[i] = base + (i+1) for i in 0..k-1
    # Start with all values, divide out small primes
    
    # For efficiency, only check up to the first failure
    # Use a list of current values
    vals = [base + i + 1 for i in range(k)]
    
    # Trial divide by small primes
    for sp in small_primes:
        if sp > p:
            break
        # Divide out sp from each val
        changed = False
        for i in range(k):
            v = vals[i]
            if v <= 1:
                continue
            while v % sp == 0:
                v //= sp
                changed = True
            vals[i] = v
        if not changed:
            # No more factors of this prime in any remaining number
            # But we still need to check larger primes
            pass
    
    # Now check each value
    run = 0
    for i in range(k):
        v = vals[i]
        if v == 1:
            run += 1
        elif v <= p:
            # v is either prime or composite with all factors <= p
            # Since we divided by all primes <= p up to sqrt, if v <= p it's fine
            # Actually we need to verify: if v is prime and <= p, ok. If composite, all factors were removed.
            # After trial division by all primes <= min(p, sqrt(original)), 
            # if v > 1 and v <= p, v must be prime (since any composite v <= p would have a factor <= sqrt(v) <= sqrt(p) < p, which we divided out)
            # Wait, not necessarily. We divided by primes up to p, so if v is composite and <= p, it has a prime factor <= sqrt(v) <= sqrt(p) < p, which should have been divided out.
            # So v <= p means v is prime or 1. Since v > 1, v is prime and <= p. Good.
            run += 1
        else:
            # v > p, need to check if all prime factors are <= p
            # v has no prime factors <= min(p, sqrt(base+k)) that we didn't divide out
            # Actually we divided by ALL primes in small_primes up to p
            # If small_primes contains all primes up to p, then v has no prime factors <= p
            # So v > 1 and v > p means v has a prime factor > p (itself or a cofactor)
            # Thus NOT p-smooth
            break
    
    return run

def construct():
    deadline = time.time() + 115  # safety margin before 120s kill
    
    best_k = 0
    best_p = 2
    
    # Precompute small primes for trial division
    # We need primes up to max p we'll test. For p up to ~10^6, primes up to 10^6 suffice for trial division
    # For larger p, we need more primes but trial division becomes the bottleneck
    # Strategy: use primes up to 100000 for quick elimination, then handle remainder
    
    SIEVE_LIMIT = 200000
    small_primes = sieve_primes(SIEVE_LIMIT)
    
    # Generate primes to search over - start with small primes
    # For small p, p^2 is small, factorization is fast
    # For large p, p^2 is large but the numbers p^2+i have structure
    
    # Phase 1: Search primes up to 1,000,000
    search_primes = sieve_primes(1000000)
    
    for p in search_primes:
        if time.time() > deadline:
            break
        if p < 2:
            continue
        
        base = p * p
        
        # Quick check: check first 8 numbers
        k_quick = check_run_smooth(base, p, 8, small_primes)
        
        # If quick check shows at least 4 consecutive smooth numbers, do full check
        if k_quick >= 4:
            k = check_run_smooth(base, p, 200, small_primes)
            
            if k > best_k:
                best_k = k
                best_p = p
    
    # Phase 2: Search larger primes using segmented sieve
    if best_k < 200 and time.time() < deadline - 20:
        # Try primes in larger ranges
        for start, end in [(1000000, 5000000), (5000000, 20000000), (20000000, 100000000)]:
            if time.time() > deadline - 10:
                break
            
            window_size = 500000
            pos = start
            while pos < end and time.time() < deadline - 10:
                sieve_end = min(pos + window_size, end)
                size = sieve_end - pos
                sieve = bytearray(b'\x01') * size
                sieve[0] = 0
                
                for sp in small_primes:
                    if sp * sp > sieve_end:
                        break
                    first = ((pos + sp - 1) // sp) * sp
                    for j in range(first - pos, size, sp):
                        sieve[j] = 0
                
                # Check candidates
                for i in range(size):
                    if time.time() > deadline - 10:
                        break
                    cand = pos + i
                    if cand < 2:
                        continue
                    if sieve[i]:
                        # Verify primality (segmented sieve may have false positives for composites with large prime factors)
                        if is_prime(cand):
                            base = cand * cand
                            k_quick = check_run_smooth(base, cand, 8, small_primes)
                            if k_quick >= 4:
                                k = check_run_smooth(base, cand, 200, small_primes)
                                if k > best_k:
                                    best_k = k
                                    best_p = cand
                pos = sieve_end
    
    if best_k == 0:
        best_p = 2
        best_k = 1
    
    return {'p': best_p, 'k': best_k}
