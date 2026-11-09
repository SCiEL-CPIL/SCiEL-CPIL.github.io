import time
import random

def construct():
    start_time = time.time()
    DEADLINE = 110  # seconds, safety margin
    
    def is_prime_mr(n):
        """Deterministic Miller-Rabin for n < 2^64"""
        if n < 2:
            return False
        if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            return True
        if n % 2 == 0:
            return False
        # Write n-1 as 2^r * d
        r, d = 0, n - 1
        while d % 2 == 0:
            r += 1
            d //= 2
        # Witnesses sufficient for n < 3,317,044,064,679,887,385,961,981
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
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
        """Pollard's rho with Brent's improvement"""
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
                    g = _gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                while True:
                    ys = (ys * ys + c) % n
                    g = _gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g
    
    def _gcd(a, b):
        while b:
            a, b = b, a % b
        return a
    
    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p"""
        if n <= 1:
            return True
        # Trial division by small primes up to min(p, 10000)
        limit = min(p, 10000)
        # Check 2
        while n % 2 == 0:
            n //= 2
        # Check odd primes up to limit
        d = 3
        while d * d <= n and d <= limit:
            while n % d == 0:
                n //= d
            d += 2
        if n == 1:
            return True
        # n is now either 1, a prime, or a composite with factors > limit
        # If n is prime, check if n <= p
        if is_prime_mr(n):
            return n <= p
        # n is composite, try to factor it
        # Use Pollard's rho to find a factor
        factor = pollard_brent(n)
        # Recursively check
        return is_p_smooth(factor, p) and is_p_smooth(n // factor, p)
    
    def largest_prime_factor_leq(n, p):
        """Return True if all prime factors of n are <= p"""
        return is_p_smooth(n, p)
    
    # Generate small primes for trial division
    def sieve(n):
        if n < 2:
            return []
        is_p = [True] * (n + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(n**0.5) + 1):
            if is_p[i]:
                for j in range(i*i, n+1, i):
                    is_p[j] = False
        return [i for i, v in enumerate(is_p) if v]
    
    small_primes = sieve(10000)
    
    best_k = 0
    best_p = 2
    
    # Check p=2 first
    # 2^2=4, 4+1=5, 5>2 so k=0 for p=2
    # Actually let's verify: we need p^2+1,...,p^2+k all p-smooth
    # For p=2: 4+1=5, 5 is prime > 2, so k=0
    
    # Search primes in increasing order
    # Use a segmented sieve or just test odd numbers
    # For efficiency, test odd numbers and use Miller-Rabin
    
    p = 3
    while p <= 1000000000:
        if time.time() - start_time > DEADLINE:
            break
        if is_prime_mr(p):
            k = 0
            base = p * p
            while k < 200:
                if time.time() - start_time > DEADLINE:
                    break
                val = base + k + 1
                if is_p_smooth(val, p):
                    k += 1
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p
                # If we hit 200, that's the max, we can stop
                if k >= 200:
                    break
        p += 2
    
    return {'p': best_p, 'k': best_k}
