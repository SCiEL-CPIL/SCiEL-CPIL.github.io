import time
import math
import random

def construct():
    deadline = time.time() + 145  # safety margin before 150s
    
    # Small primes for trial division
    small_primes = []
    d = 2
    while d <= 100000:
        is_p = True
        for q in small_primes:
            if q * q > d:
                break
            if d % q == 0:
                is_p = False
                break
        if is_p:
            small_primes.append(d)
        d += 1 if d == 2 else 2
    
    def is_prime_mr(n):
        if n < 2:
            return False
        if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            return True
        if n % 2 == 0:
            return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
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
    
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
            m = random.randint(1, n - 1)
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
                while True:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g
            # retry
    
    def factor(n, factors=None):
        if factors is None:
            factors = []
        if n <= 1:
            return factors
        if is_prime_mr(n):
            factors.append(n)
            return factors
        d = pollard_brent(n)
        factor(d, factors)
        factor(n // d, factors)
        return factors
    
    def max_prime_factor_leq(n, limit):
        """Return True if all prime factors of n are <= limit."""
        if n <= 1:
            return True
        # Quick trial division by small primes
        temp = n
        for sp in small_primes:
            if sp > limit:
                break
            if sp * sp > temp:
                break
            while temp % sp == 0:
                temp //= sp
        if temp == 1:
            return True
        if temp <= limit:
            return True
        if is_prime_mr(temp):
            return temp <= limit
        # Need to fully factor temp
        factors = factor(temp)
        for f in factors:
            if f > limit:
                return False
        return True
    
    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p."""
        if n <= 1:
            return True
        # Quick check: if n <= p, it's automatically p-smooth
        if n <= p:
            return True
        # Trial divide by small primes
        temp = n
        for sp in small_primes:
            if sp > p:
                break
            if sp * sp > temp:
                break
            while temp % sp == 0:
                temp //= sp
        if temp == 1:
            return True
        if temp <= p:
            return True
        # Check if temp is prime
        if is_prime_mr(temp):
            return temp <= p
        # Factor temp
        factors = factor(temp)
        for f in factors:
            if f > p:
                return False
        return True
    
    best_k = 0
    best_p = 2
    
    # Try known good primes first, then search
    # Start with some promising primes
    test_primes = []
    
    # Generate primes up to 10^6 for initial search
    limit = 1000000
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes_small = [i for i in range(2, limit + 1) if sieve[i]]
    
    # Also add some larger primes
    for p in primes_small:
        if time.time() > deadline:
            break
        # Check run length
        k = 0
        p2 = p * p
        while k < 200:
            if time.time() > deadline:
                break
            if is_p_smooth(p2 + k + 1, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
        if best_k >= 200:
            break
    
    # Try larger primes using random sampling
    if time.time() < deadline - 5:
        # Generate random primes in [10^6, 10^9]
        for _ in range(2000):
            if time.time() > deadline:
                break
            p = random.randint(10**6, 10**9)
            # Make it odd
            if p % 2 == 0:
                p += 1
            # Find next prime
            while not is_prime_mr(p):
                p += 2
                if p > 10**9:
                    break
            if p > 10**9:
                break
            k = 0
            p2 = p * p
            while k < 200:
                if time.time() > deadline:
                    break
                if is_p_smooth(p2 + k + 1, p):
                    k += 1
                else:
                    break
            if k > best_k:
                best_k = k
                best_p = p
            if best_k >= 200:
                break
    
    return {'p': best_p, 'k': best_k}
