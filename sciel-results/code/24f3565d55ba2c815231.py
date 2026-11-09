"""Rewritten program for the suite task; new design in the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    
    # --- Deterministic Miller-Rabin for n < 3.3e24 ---
    def is_prime_mr(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        # Write n-1 = d * 2^r
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Bases sufficient for n < 3,317,044,064,679,887,385,961,981
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
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
    
    # --- Pollard's Rho (Brent variant) ---
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
            m = 128
            g, q, r = 1, 1, 1
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
                while True:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g
    
    # --- Factorization using trial division + Pollard-Brent ---
    # Precompute small primes up to 10^5 for fast trial division
    SMALL_LIMIT = 100000
    sieve = bytearray(b'\x01') * (SMALL_LIMIT + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(SMALL_LIMIT**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    small_primes = [i for i in range(2, SMALL_LIMIT + 1) if sieve[i]]
    
    def factorize(n, limit=None):
        """Factor n, returning list of prime factors. If limit is given, only care if any factor > limit."""
        factors = []
        # Trial division by small primes
        for p in small_primes:
            if p * p > n:
                break
            if limit is not None and p > limit:
                return None  # found factor > limit, can stop
            while n % p == 0:
                factors.append(p)
                n //= p
        if n == 1:
            return factors
        # Now n is either prime or composite with factors > SMALL_LIMIT
        if is_prime_mr(n):
            if limit is not None and n > limit:
                return None
            factors.append(n)
            return factors
        # Use Pollard-Brent to factor n
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime_mr(m):
                if limit is not None and m > limit:
                    return None
                factors.append(m)
                continue
            d = pollard_brent(m)
            if d == m:
                # retry
                stack.append(m)
                continue
            stack.append(d)
            stack.append(m // d)
        return factors
    
    def max_prime_factor(v, limit):
        """Return the largest prime factor of v, or None if all factors <= limit."""
        factors = factorize(v, limit)
        if factors is None:
            return None  # some factor > limit
        return max(factors)
    
    def check_run(p, k):
        """Check if p^2+1 ... p^2+k all have prime factors <= p. Returns True/False."""
        for j in range(1, k + 1):
            val = p * p + j
            factors = factorize(val, p)
            if factors is None:
                return False
        return True
    
    def find_max_k(p):
        """Find the maximum k such that p^2+1...p^2+k are all p-smooth."""
        k = 0
        while k < 200:
            val = p * p + k + 1
            factors = factorize(val, p)
            if factors is None:
                break
            k += 1
        return k
    
    # --- Sieve for primes up to a reasonable bound ---
    # We'll search primes up to some limit. Let's start with a moderate limit
    # and increase if time permits.
    def sieve_primes(n):
        if n < 2:
            return []
        s = bytearray(b'\x01') * (n + 1)
        s[0] = s[1] = 0
        for i in range(2, int(n**0.5) + 1):
            if s[i]:
                s[i*i::i] = bytearray(len(s[i*i::i]))
        return [i for i in range(2, n + 1) if s[i]]
    
    # Start with primes up to 500,000
    PRIME_LIMIT = 500000
    primes = sieve_primes(PRIME_LIMIT)
    
    best_k = 0
    best_p = 2
    
    # First, quickly check small primes to get a baseline
    for p in primes:
        if time.time() > deadline - 5:
            break
        k = find_max_k(p)
        if k > best_k:
            best_k = k
            best_p = p
        if best_k >= 200:
            break
    
    # Now do a more targeted search: for larger primes, check runs
    # Focus on primes that might give longer runs
    # Strategy: check primes in descending order of likelihood
    # Larger primes are more likely to have shorter runs (since p^2+j grows,
    # and factors need to be <= p)
    
    # Let's also try some specific ranges
    # Check primes between 500k and 2M
    if time.time() < deadline - 10:
        PRIME_LIMIT2 = 2000000
        primes2 = sieve_primes(PRIME_LIMIT2)
        for p in primes2:
            if p <= PRIME_LIMIT:
                continue
            if time.time() > deadline - 5:
                break
            k = find_max_k(p)
            if k > best_k:
                best_k = k
                best_p = p
            if best_k >= 200:
                break
    
    # Check some larger primes up to 10M if time permits
    if time.time() < deadline - 15:
        PRIME_LIMIT3 = 10000000
        primes3 = sieve_primes(PRIME_LIMIT3)
        # Sample some primes from this range
        step = max(1, len(primes3) // 5000)
        for i in range(0, len(primes3), step):
            p = primes3[i]
            if p <= PRIME_LIMIT2:
                continue
            if time.time() > deadline - 5:
                break
            k = find_max_k(p)
            if k > best_k:
                best_k = k
                best_p = p
            if best_k >= 200:
                break
    
    # Also try some random large primes
    if time.time() < deadline - 10:
        for _ in range(1000):
            if time.time() > deadline - 5:
                break
            p = random.randint(10**7, 10**9)
            # Make it odd
            if p % 2 == 0:
                p += 1
            # Check if prime
            if not is_prime_mr(p):
                continue
            k = find_max_k(p)
            if k > best_k:
                best_k = k
                best_p = p
            if best_k >= 200:
                break
    
    # Ensure best_k is at least 1
    if best_k < 1:
        best_k = 1
        best_p = 2  # 2^2+1=5, factor 5 > 2... wait
    
    # Actually, let's verify: for p=2, p^2+1=5, largest factor 5 > 2, so k=0
    # For p=3, p^2+1=10=2*5, largest factor 5 > 3, so k=0
    # For p=5, p^2+1=26=2*13, largest factor 13 > 5, so k=0
    # For p=7, p^2+1=50=2*5^2, largest factor 5 <= 7, so k>=1
    #   p^2+2=51=3*17, largest factor 17 > 7, so k=1
    # So minimum k is 1 for p=7 at least.
    
    # Let's make sure we have a valid answer
    if best_k < 1 or best_p < 2:
        # Fallback: find a small prime with k>=1
        for p in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41]:
            k = find_max_k(p)
            if k >= 1:
                best_k = k
                best_p = p
                break
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
