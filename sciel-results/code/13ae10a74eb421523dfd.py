"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time

def _is_prime(n):
    if n < 2:
        return False
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for p in small_primes:
        if n == p:
            return True
        if n % p == 0:
            return False
    # Deterministic Miller-Rabin for n < 3.3e24
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    # Bases sufficient for n < 3,317,044,064,679,887,385,961,981
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
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

def _pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = 2 + (n - 3) * (hash(str(n)) % max(1, n - 3)) if n > 3 else 2
        c = 1 + (n - 2) * (hash(str(n) + 'c') % max(1, n - 2)) if n > 2 else 1
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
                    q = (q * abs(x - y)) % n
                g = math.gcd(q, n)
                k += m
            r *= 2
        if g == n:
            g = 1
            while g == 1:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
        if g != n:
            return g

def _factor(n, factors=None):
    if factors is None:
        factors = []
    if n == 1:
        return factors
    if _is_prime(n):
        factors.append(n)
        return factors
    d = _pollard_brent(n)
    _factor(d, factors)
    _factor(n // d, factors)
    return factors

def _largest_prime_factor(n):
    if n <= 1:
        return 1
    # Trial divide by small primes first
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
                    53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        if n % p == 0:
            while n % p == 0:
                n //= p
    if n == 1:
        return max(small_primes[:1])  # fallback, shouldn't happen for n>1
    if _is_prime(n):
        return n
    factors = _factor(n)
    return max(factors)

def _is_p_smooth(n, p):
    """Check if all prime factors of n are <= p."""
    if n <= 1:
        return True
    # Quick check: if n itself is prime and > p, fail
    if _is_prime(n) and n > p:
        return False
    # Trial divide by primes up to min(p, sqrt(n))
    # But we need to check if ANY prime factor > p exists
    # Strategy: divide out all factors <= p, see what's left
    temp = n
    # First divide by 2
    while temp % 2 == 0:
        temp //= 2
    d = 3
    limit = int(math.isqrt(temp)) + 1
    while d <= limit and d <= p:
        while temp % d == 0:
            temp //= d
            limit = int(math.isqrt(temp)) + 1
        d += 2
    if temp == 1:
        return True
    # temp > 1, check if temp is prime and <= p
    if temp <= p:
        return True
    # If temp > p, it has a prime factor > p (since we divided all factors <= p)
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 145s out of 150s, 5s safety margin
    
    # Precompute primes up to some limit for trial division
    # We'll search for primes p and check smoothness of p^2+1 ... p^2+k
    
    best_k = 0
    best_p = 2
    
    # Generate primes using a simple sieve up to some limit, then extend
    # Start with small primes, then check larger ones
    # For efficiency, let's first try a range of primes
    
    # Sieve primes up to 100000 initially
    def sieve(n):
        is_prime_arr = [True] * (n + 1)
        is_prime_arr[0] = is_prime_arr[1] = False
        for i in range(2, int(math.isqrt(n)) + 1):
            if is_prime_arr[i]:
                for j in range(i * i, n + 1, i):
                    is_prime_arr[j] = False
        return [i for i in range(2, n + 1) if is_prime_arr[i]]
    
    # Start with primes up to 10000
    primes = sieve(10000)
    
    for p in primes:
        if time.time() > deadline:
            break
        if p <= 1:
            continue
        k = 0
        # Check p^2 + 1, p^2 + 2, ..., p^2 + k
        # We want the longest run where all are p-smooth
        base = p * p
        # We can check up to 200
        for offset in range(1, 201):
            if time.time() > deadline:
                break
            n = base + offset
            if not _is_p_smooth(n, p):
                break
            k = offset
        if k > best_k:
            best_k = k
            best_p = p
            # Early exit if we hit the max
            if best_k >= 200:
                break
    
    # If we didn't find a good one in small primes, try larger primes
    # For larger p, the numbers p^2+i are larger, making smoothness harder
    # but p is larger too, so more factors are allowed
    # Let's try some larger primes
    if time.time() < deadline - 10:
        # Try primes in larger ranges
        for start in range(10003, 100000, 1000):
            if time.time() > deadline:
                break
            end = min(start + 1000, 100000)
            # Sieve a segment
            is_prime_seg = [True] * (end - start + 1)
            for i in range(2, int(math.isqrt(end)) + 1):
                if i >= start:
                    base_i = i
                else:
                    base_i = ((start + i - 1) // i) * i
                for j in range(base_i - start, end - start + 1, i):
                    if j >= 0:
                        is_prime_seg[j] = False
            for i in range(start, end):
                if time.time() > deadline:
                    break
                if i < 2 or not is_prime_seg[i - start]:
                    continue
                if not _is_prime(i):
                    continue
                k = 0
                base = i * i
                for offset in range(1, 201):
                    if time.time() > deadline:
                        break
                    n = base + offset
                    if not _is_p_smooth(n, i):
                        break
                    k = offset
                if k > best_k:
                    best_k = k
                    best_p = i
                    if best_k >= 200:
                        break
            if best_k >= 200:
                break
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
