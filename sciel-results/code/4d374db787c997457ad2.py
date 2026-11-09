import time
import random
import math

# Deterministic Miller-Rabin for n < 2^64
def is_prime(n):
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    bases = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for a in bases:
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
    """Brent's variant of Pollard's rho."""
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
    """Full factorization of n."""
    if factors is None:
        factors = []
    if n <= 1:
        return factors
    if is_prime(n):
        factors.append(n)
        return factors
    d = pollard_brent(n)
    factor(d, factors)
    factor(n // d, factors)
    return factors

def all_factors_at_most_p(n, p, small_primes):
    """Check if all prime factors of n are at most p."""
    if n <= 1:
        return True
    if n <= p:
        return True
    
    # Strip small factors using precomputed primes
    for q in small_primes:
        if q > p:
            break
        if q > n:
            break
        while n % q == 0:
            n //= q
    
    if n == 1:
        return True
    if n <= p:
        return True
    
    # n > p and no small factors remain
    if is_prime(n):
        return n <= p  # will be False since n > p
    
    # Factor the remaining part
    factors = factor(n)
    for f in factors:
        if f > p:
            return False
    return True

def sieve_primes(limit):
    if limit < 2:
        return []
    is_p = bytearray([1]) * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, limit + 1) if is_p[i]]

# Precompute small primes for trial division
_SMALL_PRIMES = sieve_primes(100000)

def segmented_sieve_primes(low, high):
    """Generate primes in [low, high] using segmented sieve."""
    if high < 2:
        return []
    limit = int(math.isqrt(high)) + 1
    base_primes = sieve_primes(limit)
    
    size = high - low + 1
    is_p = bytearray([1]) * size
    is_p[0] = 0  # low itself, handle below
    
    for p in base_primes:
        start = ((low + p - 1) // p) * p
        if start < p * p:
            start = p * p
        for j in range(start - low, size, p):
            is_p[j] = 0
    
    result = []
    for i in range(size):
        if is_p[i]:
            val = low + i
            if val >= 2:
                result.append(val)
    return result

def check_run_length(p, max_k=200):
    """Check the length of the run p^2+1, ..., p^2+k where all factors <= p."""
    p_sq = p * p
    k = 0
    while k < max_k:
        n = p_sq + k + 1
        if not all_factors_at_most_p(n, p, _SMALL_PRIMES):
            break
        k += 1
    return k

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_k = 0
    best_p = 2
    
    # Check p=2
    k = check_run_length(2)
    if k > best_k:
        best_k = k
        best_p = 2
    
    # Strategy: search primes in segments, prioritize larger primes
    # where the condition is easier to satisfy
    # We'll search in increasing order but use segmented sieve for speed
    
    # First, try a range of primes. For p around 10^5-10^7, 
    # the numbers p^2+1..p^2+200 are around 10^10-10^14,
    # and we need all prime factors <= p.
    
    # Key insight: for large p, the numbers p^2+i are likely to have
    # small prime factors. The condition "all prime factors <= p" is 
    # equivalent to: after dividing out all primes <= p, the remainder is 1.
    
    # Let's search primes in segments from small to large
    # But we can be smarter: start with moderately large primes where
    # the condition is more likely to hold for longer runs.
    
    # Segment sizes
    seg_size = 1000000  # 1 million per segment
    
    # Search from 2 up to some limit
    # Given time constraints, let's search up to about 10^7 or so
    
    search_limit = 10_000_000
    
    low = 3
    while low <= search_limit:
        if time.time() > deadline:
            break
        
        high = min(low + seg_size - 1, search_limit)
        primes_in_seg = segmented_sieve_primes(low, high)
        
        for p in primes_in_seg:
            if time.time() > deadline:
                break
            
            k = check_run_length(p)
            if k > best_k:
                best_k = k
                best_p = p
                # If we find a good run, we can try to extend the search
                # or at least note it
                
        low = high + 1
    
    # If we have time, try some larger primes
    if time.time() < deadline - 5:
        # Try primes in larger ranges
        for start in range(10_000_000, 50_000_000, 5_000_000):
            if time.time() > deadline - 5:
                break
            end = min(start + 5_000_000, 100_000_000)
            primes_in_seg = segmented_sieve_primes(start, end)
            for p in primes_in_seg:
                if time.time() > deadline - 5:
                    break
                k = check_run_length(p)
                if k > best_k:
                    best_k = k
                    best_p = p
    
    return {'p': best_p, 'k': best_k}
