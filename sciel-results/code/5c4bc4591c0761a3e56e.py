import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin

    # Precompute primes up to 10^6 for trial division
    # This allows us to quickly reduce numbers for any p
    SIEVE_LIMIT = 10**6
    def sieve_primes(limit):
        if limit < 2:
            return []
        sieve = bytearray(b'\x01') * (limit + 1)
        sieve[0] = sieve[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
        return [i for i, is_p in enumerate(sieve) if is_p]

    trial_primes = sieve_primes(SIEVE_LIMIT)
    
    # Deterministic Miller-Rabin for n < 2^64
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        for p in MR_BASES:
            if n % p == 0:
                return n == p
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in MR_BASES:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
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
        if n % 5 == 0:
            return 5
        if n % 7 == 0:
            return 7
        if n % 11 == 0:
            return 11
        if n % 13 == 0:
            return 13
        if n % 17 == 0:
            return 17
        if n % 19 == 0:
            return 19
        if n % 23 == 0:
            return 23
        if n % 29 == 0:
            return 29
        if n % 31 == 0:
            return 31
        if n % 37 == 0:
            return 37
        while True:
            y = random.randrange(1, n)
            c = random.randrange(1, n)
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

    def max_prime_factor(n):
        """Returns the largest prime factor of n."""
        if n <= 1:
            return 1
        factors = []
        # Trial division by small primes
        for p in trial_primes:
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    n //= p
                factors.append(p)
        if n > 1:
            if is_prime(n):
                factors.append(n)
            else:
                # n is composite and has no factors in trial_primes
                # Factor n
                d = pollard_brent(n)
                f1 = max_prime_factor(d)
                f2 = max_prime_factor(n // d)
                factors.append(f1)
                factors.append(f2)
        return max(factors) if factors else 1

    def is_p_smooth(n, p):
        """Checks if all prime factors of n are <= p."""
        if n <= p:
            return True
        # We need to check if max_prime_factor(n) <= p
        # To speed up, we can first divide by small primes
        # But max_prime_factor already does trial division.
        # However, calling max_prime_factor for every n is slow.
        # Let's integrate the check.
        
        rem = n
        # Trial divide by primes <= min(p, SIEVE_LIMIT)
        upper = min(p, SIEVE_LIMIT)
        for q in trial_primes:
            if q > upper:
                break
            if q * q > rem:
                break
            if rem % q == 0:
                while rem % q == 0:
                    rem //= q
        
        if rem == 1:
            return True
        if rem <= p:
            return True
        
        # rem > p
        # rem is either prime or composite
        if is_prime(rem):
            return rem <= p
        
        # rem is composite
        # Check if rem has any factor <= p
        # We already divided by primes <= upper.
        # If p <= SIEVE_LIMIT, we divided by all primes <= p.
        # So if rem > 1 and p <= SIEVE_LIMIT, rem has no factors <= p.
        # Thus rem is not p-smooth.
        if p <= SIEVE_LIMIT:
            return False
        
        # If p > SIEVE_LIMIT, we only divided by primes <= SIEVE_LIMIT.
        # rem might have factors in (SIEVE_LIMIT, p].
        # We need to factor rem and check max factor.
        # rem is < n ~ p^2.
        d = pollard_brent(rem)
        return is_p_smooth(d, p) and is_p_smooth(rem // d, p)

    def check_run(p, max_k=200):
        k = 0
        base = p * p
        while k < max_k:
            n = base + k + 1
            if not is_p_smooth(n, p):
                break
            k += 1
        return k

    best_k = 0
    best_p = 2

    # Generate primes for search
    # We search primes up to 10^7 first, then sample larger.
    search_limit = 10**7
    primes_to_check = sieve_primes(search_limit)
    
    for p in primes_to_check:
        if time.time() > deadline:
            break
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p
            
    # If time remains, sample larger primes
    if time.time() < deadline - 10:
        ranges = [(10**7, 10**8), (10**8, 10**9)]
        for low, high in ranges:
            if time.time() > deadline:
                break
            attempts = 0
            while attempts < 100 and time.time() < deadline:
                x = random.randrange(low, high)
                if x % 2 == 0:
                    x += 1
                while not is_prime(x):
                    x += 2
                    if x > high:
                        break
                if x > high:
                    break
                attempts += 1
                k = check_run(x)
                if k > best_k:
                    best_k = k
                    best_p = x

    return {'p': best_p, 'k': best_k}
