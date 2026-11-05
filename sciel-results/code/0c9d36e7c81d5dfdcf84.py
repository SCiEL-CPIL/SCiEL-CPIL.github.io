import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit

    # Precompute primes up to 10^6 for trial division
    P_MAX = 1000000
    is_prime_sieve = bytearray(b'\x01') * (P_MAX + 1)
    is_prime_sieve[0] = is_prime_sieve[1] = 0
    for i in range(2, int(math.isqrt(P_MAX)) + 1):
        if is_prime_sieve[i]:
            for j in range(i*i, P_MAX + 1, i):
                is_prime_sieve[j] = 0
    primes = [i for i in range(2, P_MAX + 1) if is_prime_sieve[i]]
    primes_set = set(primes)

    # Deterministic Miller-Rabin for n < 3.2 * 10^18
    def is_prime_mr(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        # Check small primes
        for p in [5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == p:
                return True
            if n % p == 0:
                return False
        # Write n-1 as 2^r * d
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Deterministic witnesses for n < 3.2 * 10^18
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
                        q = (q * abs(x - y)) % n
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
        """Recursively factor n into prime factors."""
        if factors is None:
            factors = []
        if n <= 1:
            return factors
        if is_prime_mr(n):
            factors.append(n)
            return factors
        # Try small primes first
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                factor(p, factors)
                factor(n // p, factors)
                return factors
        # Use Pollard-Brent
        d = pollard_brent(n)
        factor(d, factors)
        factor(n // d, factors)
        return factors

    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p."""
        if n <= 1:
            return True
        if n <= p:
            return True
        # Quick check: if n has a factor > p, it's not p-smooth
        # Factor n and check max factor
        facs = factor(n)
        for f in facs:
            if f > p:
                return False
        return True

    def find_max_k(p, max_k=200):
        """Find max k such that p^2+1...p^2+k are all p-smooth."""
        k = 0
        p2 = p * p
        for j in range(1, max_k + 1):
            if not is_p_smooth(p2 + j, p):
                break
            k += 1
        return k

    def sieve_primes_up_to(n):
        """Generate primes up to n using sieve."""
        if n < 2:
            return []
        sieve = bytearray(b'\x01') * (n + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(math.isqrt(n)) + 1):
            if sieve[i]:
                for j in range(i*i, n + 1, i):
                    sieve[j] = 0
        return [i for i in range(2, n + 1) if sieve[i]]

    best_k = 1
    best_p = 2

    # Strategy: Search primes in ranges, prioritizing larger p where
    # p-smoothness is more likely for longer runs.
    # Use a mix of deterministic and random search.
    
    # First, try known good ranges with moderate primes
    # p^2 + j for j up to 200, we need all factors <= p
    # For larger p, the numbers p^2+j are ~p^2, so they have factors
    # that could be up to p^2. But if all factors are <= p, that's p-smooth.
    
    # Search strategy:
    # 1. Check primes in [1000, 100000] with full factorization
    # 2. If time remains, check larger primes with quick screening
    
    # Phase 1: Check primes from 1000 to 100000 (step through all primes)
    search_primes = [p for p in primes if 1000 <= p <= 100000]
    
    for p in search_primes:
        if time.time() - start_time > DEADLINE:
            break
        # Quick screen: check p^2+1 first
        if not is_p_smooth(p * p + 1, p):
            continue
        # Full check
        k = find_max_k(p)
        if k > best_k:
            best_k = k
            best_p = p
        # If we found k >= 10, this is promising; continue
    
    # Phase 2: If time remains, search larger primes
    if time.time() - start_time < DEADLINE - 30:
        # Check primes in [100000, 1000000]
        large_primes = [p for p in primes if 100000 <= p <= 1000000]
        # Sample these primes (can't check all)
        step = max(1, len(large_primes) // 500)  # Check ~500 primes
        for p in large_primes[::step]:
            if time.time() - start_time > DEADLINE - 5:
                break
            if not is_p_smooth(p * p + 1, p):
                continue
            k = find_max_k(p)
            if k > best_k:
                best_k = k
                best_p = p
    
    # Phase 3: If time remains, try very large primes with random sampling
    if time.time() - start_time < DEADLINE - 60:
        # Generate primes up to 10^7
        P_MAX2 = 10000000
        sieve2 = bytearray(b'\x01') * (P_MAX2 + 1)
        sieve2[0] = sieve2[1] = 0
        for i in range(2, int(math.isqrt(P_MAX2)) + 1):
            if sieve2[i]:
                for j in range(i*i, P_MAX2 + 1, i):
                    sieve2[j] = 0
        large_primes2 = [i for i in range(1000000, P_MAX2 + 1) if sieve2[i]]
        
        # Randomly sample ~200 primes
        random.seed(42)
        sample = random.sample(large_primes2, min(200, len(large_primes2)))
        for p in sample:
            if time.time() - start_time > DEADLINE - 5:
                break
            if not is_p_smooth(p * p + 1, p):
                continue
            k = find_max_k(p)
            if k > best_k:
                best_k = k
                best_p = p

    return {'p': best_p, 'k': best_k}
