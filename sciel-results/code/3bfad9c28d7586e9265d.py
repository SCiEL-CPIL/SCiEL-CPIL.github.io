import time
import math
import random
import sys

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s

    # Deterministic Miller-Rabin for n < 2^64 (covers p^2+k for p<=10^9, k<=200)
    # Witnesses sufficient for n < 3,317,044,064,679,887,385,961,981
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        # Small prime checks
        small_primes = [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        # Miller-Rabin
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in MR_BASES:
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

    # Pollard-Brent factorization
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
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
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
            if g != n:
                return g

    def factorize(n):
        """Return list of prime factors of n (with multiplicity)."""
        if n <= 1:
            return []
        factors = []
        # Trial division by small primes first
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for p in small_primes:
            while n % p == 0:
                factors.append(p)
                n //= p
        if n == 1:
            return factors
        # Use Pollard-Brent for the rest
        stack = [n]
        while stack:
            x = stack.pop()
            if x == 1:
                continue
            if is_prime(x):
                factors.append(x)
            else:
                d = pollard_brent(x)
                if d == x:
                    # Shouldn't happen, but fallback
                    stack.append(x)
                    break
                stack.append(d)
                stack.append(x // d)
        return factors

    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p."""
        if n <= 1:
            return True
        # Quick check: if n <= p, it's trivially p-smooth
        if n <= p:
            return True
        # Factor n and check max factor
        # First try trial division by small primes
        remaining = n
        for q in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199]:
            if q > p:
                break
            while remaining % q == 0:
                remaining //= q
        if remaining == 1:
            return True
        if remaining <= p:
            return True
        # remaining > p. Need to factor it fully.
        # Use Pollard-Brent
        stack = [remaining]
        while stack:
            x = stack.pop()
            if x == 1:
                continue
            if x > p:
                # x has a factor > p (at least x itself if prime, or some factor)
                # But x might be composite with all factors <= p... no wait, we removed small primes.
                # Actually we need to be careful. Let's just factor x fully.
                pass
            if is_prime(x):
                if x > p:
                    return False
            else:
                d = pollard_brent(x)
                if d == x:
                    # Fallback: trial division
                    found = False
                    limit = int(math.isqrt(x))
                    for d2 in range(2, limit + 1, 2):
                        if x % d2 == 0:
                            if d2 > p:
                                return False
                            stack.append(d2)
                            stack.append(x // d2)
                            found = True
                            break
                    if not found:
                        # x is prime
                        if x > p:
                            return False
                else:
                    stack.append(d)
                    stack.append(x // d)
        return True

    def find_max_k(p, max_k=200):
        """Find the maximum k such that p^2+1..p^2+k are all p-smooth."""
        k = 0
        p2 = p * p
        for j in range(1, max_k + 1):
            if not is_p_smooth(p2 + j, p):
                break
            k += 1
        return k

    best_k = 0
    best_p = 2

    # Check known primes first - these are cheap to test
    known_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499]

    for p in known_primes:
        if time.time() - start_time > DEADLINE:
            break
        k = find_max_k(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Now search larger primes. Strategy: sample primes in increasing ranges.
    # For each prime p, the cost of find_max_k depends on how far we get.
    # Most primes will fail at j=1 or j=2, so they're cheap.
    # We want to find primes where p^2+1, p^2+2, ... are all p-smooth.
    
    # Use a sieve-like approach: generate primes in ranges and test them.
    # For efficiency, we'll check many small candidates quickly.
    
    def sieve_primes(lo, hi):
        """Generate primes in [lo, hi] using segmented sieve."""
        if hi < 2:
            return []
        lo = max(lo, 2)
        size = hi - lo + 1
        is_p = bytearray(b'\x01') * size
        # Sieve by primes up to sqrt(hi)
        limit = int(math.isqrt(hi))
        # Generate small primes up to limit
        small_limit = max(limit, 100)
        small_sieve = bytearray(b'\x01') * (small_limit + 1)
        small_sieve[0] = small_sieve[1] = 0
        for i in range(2, int(math.isqrt(small_limit)) + 1):
            if small_sieve[i]:
                for j in range(i*i, small_limit + 1, i):
                    small_sieve[j] = 0
        small_primes = [i for i in range(2, small_limit + 1) if small_sieve[i]]
        
        for p in small_primes:
            if p > limit:
                break
            # Mark multiples of p in [lo, hi]
            start = ((lo + p - 1) // p) * p
            if start == p:
                start += p
            for j in range(start, hi + 1, p):
                is_p[j - lo] = 0
        
        return [lo + i for i in range(size) if is_p[i]]

    # Search in increasing ranges, spending time proportional to range size
    # but with a time budget
    search_ranges = [
        (500, 5000),
        (5000, 20000),
        (20000, 100000),
        (100000, 500000),
        (500000, 2000000),
        (2000000, 10000000),
        (10000000, 50000000),
        (50000000, 200000000),
        (200000000, 1000000000),
    ]

    for lo, hi in search_ranges:
        if time.time() - start_time > DEADLINE:
            break
        # Get primes in this range
        primes = sieve_primes(lo, hi)
        # Test each prime
        for p in primes:
            if time.time() - start_time > DEADLINE:
                break
            k = find_max_k(p)
            if k > best_k:
                best_k = k
                best_p = p

    # Do a denser search around the best found
    if best_p > 1000 and time.time() - start_time < DEADLINE - 10:
        center = best_p
        radius = min(50000, int(math.sqrt(center)) * 10)
        lo = max(2, center - radius)
        hi = min(10**9, center + radius)
        primes = sieve_primes(lo, hi)
        for p in primes:
            if time.time() - start_time > DEADLINE:
                break
            if p == best_p:
                continue
            k = find_max_k(p)
            if k > best_k:
                best_k = k
                best_p = p

    # Ensure we have at least k=1
    if best_k < 1:
        best_k = 1
        best_p = 2

    return {'p': best_p, 'k': best_k}
