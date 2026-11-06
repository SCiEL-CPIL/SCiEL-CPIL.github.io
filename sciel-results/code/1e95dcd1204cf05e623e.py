import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 118.0  # safety margin before 120s limit (with 150s kill)

    # Miller-Rabin primality test (deterministic for n < 3.3e24)
    def is_prime(n):
        if n < 2:
            return False
        if n in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            return True
        if n % 2 == 0 or n % 3 == 0 or n % 5 == 0:
            return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Bases for deterministic MR for n < 3,317,044,064,679,887,385,961,981
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

    # Pollard's Rho with Brent's cycle detection
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            x = random.randint(2, n - 1)
            y = x
            d = 1
            c = random.randint(1, n - 1)
            m = 128
            g = q = r = 1
            while d == 1:
                x = y
                for _ in range(r):
                    y = (y * y + c) % n
                k = 0
                while k < r and d == 1:
                    ys = y
                    for _ in range(min(m, r - k)):
                        y = (y * y + c) % n
                        q = (q * abs(x - y)) % n
                    g = math.gcd(q, n)
                    k += m
                r *= 2
            if d == n:
                d = 1
                while d == 1:
                    ys = (ys * ys + c) % n
                    d = math.gcd(abs(x - ys), n)
            if d != n:
                return d

    # Factorization using trial division by small primes, then Pollard's Rho
    def factor(n, small_primes=None, max_trial=10000):
        factors = []
        if n < 2:
            return factors
        if is_prime(n):
            return [n]
        
        # Trial division by small primes
        trial_limit = 10000
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
            if p * p > n:
                break
            while n % p == 0:
                factors.append(p)
                n //= p
        if n == 1:
            return factors
        if is_prime(n):
            factors.append(n)
            return factors
        
        # Pollard's Rho for the remaining composite
        def _factor_recursive(m):
            if m == 1:
                return []
            if is_prime(m):
                return [m]
            d = pollard_rho(m)
            return _factor_recursive(d) + _factor_recursive(m // d)
        
        factors.extend(_factor_recursive(n))
        return factors

    # Check if n is p-smooth (all prime factors <= p)
    def is_p_smooth(n, p):
        if n <= p:
            return True
        # Trial divide by small primes first for speed
        for q in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
            while n % q == 0:
                n //= q
                if n == 1:
                    return True
                if n < p:
                    # All remaining factors are < p, so they're <= p
                    return True
        if n == 1:
            return True
        if n < p:
            return True
        if is_prime(n):
            return n <= p
        # Factor the remaining part
        factors = factor(n)
        return all(f <= p for f in factors)

    # Check run length for a given prime p
    def check_run(p, k_max=200):
        k = 0
        p_sq = p * p
        for i in range(1, k_max + 1):
            if time.time() > DEADLINE - 2:
                break
            n = p_sq + i
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        return k

    best_k = 0
    best_p = 2

    # Phase 1: Check small primes up to 100000 with fast trial division
    # For small p, trial division is very fast
    small_primes = []
    def sieve(limit):
        is_p = bytearray([1]) * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                is_p[i*i:limit+1:i] = bytearray(len(range(i*i, limit+1, i)))
        return [i for i in range(2, limit+1) if is_p[i]]
    
    small_primes = sieve(100000)
    
    def is_p_smooth_small(n, p):
        """Fast p-smooth check for small p using trial division by primes up to p"""
        m = n
        for q in small_primes:
            if q > p or q * q > m:
                break
            while m % q == 0:
                m //= q
                if m == 1:
                    return True
        if m == 1:
            return True
        if m < p:
            return True
        # m might be prime or composite with factors > p
        # If m is prime, check m <= p
        # If m is composite, it has a factor <= sqrt(m) <= sqrt(n) ~ p
        # So if all factors are > p, then m > p^2, but m < n ~ p^2, contradiction
        # Actually if m has a factor <= p, we would have divided it out
        # So m is either 1, prime > p, or composite with all factors > p
        # If composite with all factors > p, then m > p^2 > n, impossible
        # So m must be prime
        return m <= p

    def check_run_small(p, k_max=200):
        k = 0
        p_sq = p * p
        for i in range(1, k_max + 1):
            if time.time() > DEADLINE - 2:
                break
            n = p_sq + i
            if is_p_smooth_small(n, p):
                k += 1
            else:
                break
        return k

    # Search small primes
    for p in small_primes:
        if time.time() > DEADLINE - 5:
            break
        k = check_run_small(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Random search for larger primes
    # For larger p, use Pollard's Rho based factorization
    random.seed(42)
    
    # Try some structured candidates first
    # Primes around specific ranges
    candidate_ranges = [
        (100000, 500000),
        (500000, 1000000),
        (1000000, 5000000),
        (5000000, 10000000),
        (10000000, 50000000),
        (50000000, 100000000),
        (100000000, 1000000000),
    ]
    
    for lo, hi in candidate_ranges:
        if time.time() > DEADLINE - 5:
            break
        # Try a few random primes in this range
        attempts = 0
        while time.time() < DEADLINE - 5 and attempts < 50:
            attempts += 1
            # Pick random odd number
            p = random.randint(lo, hi)
            if p % 2 == 0:
                p += 1
            # Find next prime
            while not is_prime(p) and time.time() < DEADLINE - 5:
                p += 2
            if p > 10**9 or p <= 1:
                continue
            k = check_run(p)
            if k > best_k:
                best_k = k
                best_p = p

    # Also try near powers of 2 and other structured values
    structured_candidates = []
    for exp in range(17, 30):
        val = 1 << exp
        # Find primes near this value
        for delta in range(-1000, 1001, 2):
            candidate = val + delta
            if 2 <= candidate <= 10**9 and is_prime(candidate):
                structured_candidates.append(candidate)
                break
    
    for p in structured_candidates:
        if time.time() > DEADLINE - 5:
            break
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    return {'p': best_p, 'k': best_k}
