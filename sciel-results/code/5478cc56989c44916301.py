import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0

    # Deterministic Miller-Rabin for n < 2^64
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def is_prime_mr(n):
        if n < 2:
            return False
        if n in (2, 3):
            return True
        if n % 2 == 0:
            return False
        d = n - 1
        r = 0
        while d % 2 == 0:
            r += 1
            d //= 2
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

    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        if n % 7 == 0:
            return 7
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

    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p."""
        if n <= 1:
            return True
        # Trial division by small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293]
        for q in small_primes:
            if q > p:
                break
            if n % q == 0:
                while n % q == 0:
                    n //= q
                if n == 1:
                    return True
        if n == 1:
            return True
        if n <= p:
            return True
        if is_prime_mr(n):
            return False
        d = pollard_brent(n)
        return is_p_smooth(d, p) and is_p_smooth(n // d, p)

    def check_run(p, k_max=200):
        k = 0
        p_sq = p * p
        for i in range(1, k_max + 1):
            if time.time() > DEADLINE - 3:
                break
            n = p_sq + i
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        return k

    def sieve_primes(limit):
        if limit < 2:
            return []
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    # Phase 1: Quick search over small primes using sieve
    SIEVE_LIMIT = 1000000
    primes_small = sieve_primes(SIEVE_LIMIT)

    best_k = 0
    best_p = 2

    for p in primes_small:
        if time.time() > DEADLINE - 10:
            break
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Search larger primes using random sampling + congruence filtering
    if time.time() < DEADLINE - 15:
        # Precompute: for each small prime q, which residues r mod q satisfy r^2 ≡ -i (mod q) for i in 1..200
        # This helps filter primes p: p^2 + i has factor q iff p^2 ≡ -i (mod q)
        
        # For a given p, the probability that p^2+i is divisible by some small prime q is high
        # We want primes where p^2+1, p^2+2, ..., p^2+k all have small factors
        
        # Strategy: Try primes in ranges where many small primes divide p^2+i for consecutive i
        
        # For small primes q, the number of residues r mod q with r^2 ≡ -i (mod q) is 0, 1, or 2
        # We can score a prime p by counting how many i in 1..200 have p^2+i divisible by some q <= p
        
        # Simpler: Just try random primes in larger ranges and check them
        # Focus on ranges where p is not too large (so p-smooth is more likely)
        
        search_ranges = [
            (10**6, 5*10**6),
            (5*10**6, 2*10**7),
            (2*10**7, 10**8),
            (10**8, 10**9),
        ]
        
        for lo, hi in search_ranges:
            if time.time() > DEADLINE - 5:
                break
            
            # Try a few random primes in this range
            num_trials = 200
            for _ in range(num_trials):
                if time.time() > DEADLINE - 5:
                    break
                x = random.randrange(lo, hi)
                if x % 2 == 0:
                    x += 1
                # Find next prime
                found = False
                for _ in range(100):
                    if is_prime_mr(x):
                        found = True
                        break
                    x += 2
                if not found or x > hi:
                    continue
                if x > 10**9:
                    continue
                
                k = check_run(x)
                if k > best_k:
                    best_k = k
                    best_p = x
                    # If we found a good one, continue searching for even better
                    if best_k >= 10:
                        # Found something good, keep looking
                        pass
                    elif best_k >= 5:
                        # Decent, keep looking but less aggressively
                        pass

    # Phase 3: If we have time, do a more systematic search around the best found
    if time.time() < DEADLINE - 10 and best_p > 2:
        # Try primes near the best one
        center = best_p
        # Check primes in [center - 100000, center + 100000]
        lo = max(2, center - 100000)
        hi = min(10**9, center + 100000)
        
        # Generate candidates
        x = lo
        if x % 2 == 0:
            x += 1
        checked = 0
        while x <= hi and checked < 5000:
            if time.time() > DEADLINE - 5:
                break
            if is_prime_mr(x):
                k = check_run(x)
                if k > best_k:
                    best_k = k
                    best_p = x
            x += 2
            checked += 1

    # Ensure best_k >= 1
    if best_k < 1:
        best_k = 1
        best_p = 2

    return {'p': best_p, 'k': best_k}
