import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Deterministic Miller-Rabin for n < 3.3e24 (covers our range)
    _MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def is_prime(n):
        if n < 2:
            return False
        for p in _MR_BASES:
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
                x = x * x % n
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
            y = random.randrange(2, n - 1)
            c = random.randrange(1, n - 1)
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
                while True:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g
    
    def factorize(n):
        """Return set of distinct prime factors of n."""
        factors = set()
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime(m):
                factors.add(m)
                continue
            d = pollard_brent(m)
            stack.append(d)
            stack.append(m // d)
        return factors
    
    def run_length(p, max_k=200):
        """Return the length of the run starting at p^2+1 where all factors <= p."""
        p_sq = p * p
        for i in range(1, max_k + 1):
            n = p_sq + i
            # Quick check: if n is prime and n > p, it fails immediately
            if is_prime(n) and n > p:
                return i - 1
            # Full factorization check
            for f in factorize(n):
                if f > p:
                    return i - 1
        return max_k
    
    # Sieve for candidate primes
    def sieve(limit):
        is_p = bytearray([1]) * (limit + 1)
        is_p[0] = 0
        is_p[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                is_p[i*i:limit+1:i] = bytearray(len(range(i*i, limit+1, i)))
        return [i for i, val in enumerate(is_p) if val]
    
    # Generate candidate primes
    limit = 5000000
    primes = sieve(limit)
    
    best_k = 0
    best_p = 2
    
    # Strategy: check primes in order, but prioritize larger primes which are more likely
    # to have longer runs (since more small primes are available as factors)
    # Actually, let's check all primes but with a time budget
    
    # First pass: check all primes quickly
    for p in primes:
        if time.time() > deadline:
            break
        
        k = run_length(p, 200)
        if k > best_k:
            best_k = k
            best_p = p
            
    # If we have time, try some larger primes (up to 10^9)
    # Use a larger sieve or random sampling
    if time.time() < deadline - 10:
        # Try primes in a higher range
        # Generate primes up to 10^7 for a second pass on higher range
        limit2 = 10**7
        primes2 = sieve(limit2)
        # Only check primes above limit
        for p in primes2:
            if time.time() > deadline:
                break
            if p <= limit:
                continue
            k = run_length(p, 200)
            if k > best_k:
                best_k = k
                best_p = p
    
    # If still have time, try random primes up to 10^9
    if time.time() < deadline - 5:
        random.seed(42)
        # Try some random large primes
        for _ in range(200):
            if time.time() > deadline:
                break
            # Generate random number in [10^7, 10^9] and find next prime
            candidate = random.randint(10**7, 10**9)
            # Find next prime
            while not is_prime(candidate):
                candidate += 1
            if candidate > 10**9:
                continue
            # Skip if we've likely checked it (heuristic)
            if candidate < limit2:
                continue
            k = run_length(candidate, 200)
            if k > best_k:
                best_k = k
                best_p = candidate
    
    return {'p': best_p, 'k': best_k}
