"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import math
import random

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    
    # Precompute small primes up to 10^6 for fast trial division
    SMALL_LIMIT = 10**6
    sieve = bytearray(b'\x01') * (SMALL_LIMIT + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(SMALL_LIMIT**0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    small_primes = [i for i in range(2, SMALL_LIMIT + 1) if sieve[i]]
    
    # Deterministic Miller-Rabin for n < 2^64
    def is_prime_mr(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return n in (2, 3)
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Witnesses for n < 2^64
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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
    
    # Pollard's rho (Brent's variant)
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
                while True:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g
    
    # Full factorization: returns list of prime factors
    def factorize(n):
        if n <= 1:
            return []
        if is_prime_mr(n):
            return [n]
        d = pollard_brent(n)
        return factorize(d) + factorize(n // d)
    
    # Check if n is p-smooth using small primes first, then full factorization
    def is_p_smooth(n, p):
        if n <= 1:
            return True
        # Try small primes up to min(p, SMALL_LIMIT)
        for sp in small_primes:
            if sp > p:
                break
            while n % sp == 0:
                n //= sp
        if n == 1:
            return True
        # Remaining factor: must be <= p and prime (or product of primes <= p)
        # If n > p, it cannot be p-smooth (since any prime factor of n > p)
        # Actually we need to check if all prime factors of n are <= p
        if n > p:
            # Could still be p-smooth if n = q1*q2*... where all qi <= p
            # But if n > p and n is prime, then n > p so not p-smooth
            # We need full factorization
            factors = factorize(n)
            return all(f <= p for f in factors)
        else:
            # n <= p, so if n is prime, it's <= p, good
            # If n is composite, its factors are < n <= p, so all <= p
            return True
    
    # Quick check: how many consecutive values starting from p^2+1 are p-smooth?
    # Uses only small prime trial division for speed
    def quick_smooth_run(p, max_k=200):
        """Returns the length of the run of p-smooth numbers starting at p^2+1.
        Uses only small primes; if remainder > 1 and > p, conservatively stops."""
        k = 0
        p_sq = p * p
        for i in range(1, max_k + 1):
            n = p_sq + i
            if n <= 1:
                continue
            # Trial divide by small primes <= p
            for sp in small_primes:
                if sp > p:
                    break
                if sp * sp > n:
                    break
                while n % sp == 0:
                    n //= sp
            if n == 1:
                k += 1
            elif n <= p:
                # n is a prime factor <= p, so p-smooth
                k += 1
            else:
                # n > p: could be prime > p (fail) or composite with factors <= p
                # For quick check, if n is prime it fails. If composite, might pass.
                # Conservatively: check if n is prime
                if is_prime_mr(n):
                    break
                else:
                    # Composite and > p: need to check if all factors <= p
                    # This is expensive; for quick check, assume fail to be safe
                    # But this might miss valid cases. Let's do a partial check:
                    # Try dividing by small primes again (we already did)
                    # If after trial division n > p and composite, factors could be <= p
                    # Example: n = 99991 * 99989, both < p=10^6. But we already divided by small primes.
                    # Actually we already divided by all small primes <= p. So if n > p after that,
                    # and n is composite, its prime factors are all > SMALL_LIMIT (since we divided by all <= SMALL_LIMIT)
                    # Wait no: we divided by small primes up to min(p, SMALL_LIMIT). If p > SMALL_LIMIT,
                    # we only divided by primes up to SMALL_LIMIT. So n could have prime factors between SMALL_LIMIT and p.
                    # This is getting complex. For the quick check, let's just break.
                    break
        return k
    
    # Full verification: verify that p^2+1 through p^2+k are all p-smooth
    def verify_run(p, k):
        p_sq = p * p
        for i in range(1, k + 1):
            n = p_sq + i
            if not is_p_smooth(n, p):
                return False
        return True
    
    # Generate primes up to a limit using the sieve
    def primes_up_to(limit):
        if limit < 2:
            return []
        s = bytearray(b'\x01') * (limit + 1)
        s[0] = s[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if s[i]:
                s[i*i::i] = bytearray(len(s[i*i::i]))
        return [i for i in range(2, limit + 1) if s[i]]
    
    best_k = 0
    best_p = 2
    
    # Strategy: 
    # 1. First, search small primes (up to 10^5) thoroughly
    # 2. Then, for larger primes, use quick check to find candidates, then verify
    
    # Phase 1: Small primes up to 10^5
    small_search_limit = 100000
    primes_small = primes_up_to(small_search_limit)
    
    for p in primes_small:
        if time.time() > deadline:
            break
        k = quick_smooth_run(p, 200)
        if k > best_k:
            # Verify with full factorization
            if verify_run(p, k):
                best_k = k
                best_p = p
    
    # Phase 2: Larger primes - sample strategically
    # For larger p, p^2 is huge. The probability that p^2+i is p-smooth decreases.
    # Use random sampling of primes in larger ranges
    
    # Generate primes in larger ranges by testing odd numbers
    def is_prime_test(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return n in (2, 3)
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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
    
    # Search in chunks for larger primes
    # Focus on primes where p^2 + i might be smooth
    # Known: small primes like 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37 often give good runs
    
    # Try specific ranges with higher density of smooth numbers
    # For p around 10^3 to 10^6, do denser search
    for start in range(100000, min(10**6, 10**7), 50000):
        if time.time() > deadline:
            break
        end = start + 50000
        for candidate in range(start, end, 2):
            if time.time() > deadline:
                break
            if not is_prime_test(candidate):
                continue
            # Quick check
            k = quick_smooth_run(candidate, 200)
            if k > best_k:
                if verify_run(candidate, k):
                    best_k = k
                    best_p = candidate
    
    # Phase 3: Random sampling of larger primes
    if time.time() < deadline:
        attempts = 0
        while time.time() < deadline and attempts < 5000:
            attempts += 1
            # Random prime candidate
            if random.random() < 0.5:
                # Between 10^6 and 10^7
                candidate = random.randint(10**6, 10**7)
            elif random.random() < 0.5:
                # Between 10^7 and 10^8
                candidate = random.randint(10**7, 10**8)
            else:
                # Between 10^8 and 10^9
                candidate = random.randint(10**8, 10**9)
            
            # Make odd
            if candidate % 2 == 0:
                candidate += 1
            
            # Find next prime
            for _ in range(100):
                if is_prime_test(candidate):
                    break
                candidate += 2
            else:
                continue
            
            # Quick check
            k = quick_smooth_run(candidate, 200)
            if k > best_k:
                if verify_run(candidate, k):
                    best_k = k
                    best_p = candidate
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
