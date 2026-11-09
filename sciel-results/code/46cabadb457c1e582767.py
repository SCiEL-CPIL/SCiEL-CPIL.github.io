import time
import math
import random

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s limit

    # Deterministic Miller-Rabin for n < 3.3e24
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

    # Pollard's Brent cycle detection
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

    # Sieve primes up to limit
    def sieve_primes(limit):
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                step = i
                start = i * i
                is_p[start::step] = bytearray(len(is_p[start::step]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    # Precompute primes up to a large limit for segmented sieve
    # We need primes up to p for the segmented sieve over [p^2+1, p^2+200]
    # But p can be up to 10^9, so we can't precompute all primes up to p.
    # Instead, we use a two-phase approach:
    # Phase 1: Use precomputed small primes to sieve the interval
    # Phase 2: For remaining factors, use Pollard-Brent

    # Precompute primes up to 10^6 (covers most small factors)
    SIEVE_LIMIT = 1000000
    small_primes = sieve_primes(SIEVE_LIMIT)

    def check_run_segmented(p, k_max=200):
        """Check p-smoothness of [p^2+1, p^2+k_max] using segmented sieve."""
        base = p * p
        # Create array to track remaining value for each position
        # remaining[i] = (base + i) after removing all small prime factors
        remaining = [base + i for i in range(1, k_max + 1)]
        
        # Sieve out all small primes <= min(p, SIEVE_LIMIT)
        for q in small_primes:
            if q > p:
                break
            # Find first multiple of q in [base+1, base+k_max]
            # We need (base + i) % q == 0 for i in [1, k_max]
            # i.e., base + i ≡ 0 (mod q) => i ≡ -base (mod q)
            r = (-base) % q
            if r == 0:
                start_i = q  # i = q, 2q, ... but must be >= 1
                # Actually i ≡ 0 (mod q), so i = q, 2q, ...
                # But i must be in [1, k_max]
                start_i = q
            else:
                start_i = r
                if start_i < 1:
                    start_i += q
            
            for i in range(start_i, k_max + 1, q):
                if i >= 1 and i <= k_max:
                    while remaining[i-1] % q == 0:
                        remaining[i-1] //= q
        
        # Now check if all remaining values are 1 or prime <= p
        k = 0
        for i in range(1, k_max + 1):
            n = remaining[i-1]
            if n == 1:
                k = i
                continue
            # n might be prime or composite with factors > SIEVE_LIMIT
            # Check if n is prime and <= p
            if n <= p:
                # n is a factor <= p, so it's fine (could be prime or 1)
                # But we need to verify n itself has no factor > p
                # Since n <= p, all its factors are <= p
                k = i
                continue
            # n > p, need to check if n is prime (bad) or composite with factors <= p
            # If n is prime, it's > p, so not p-smooth
            if is_prime_mr(n):
                break
            # n is composite, try to factor it
            # If all factors are <= p, it's p-smooth
            factor = pollard_brent(n)
            # Check if factor and n//factor are both p-smooth
            # Since n = base + i and we've removed small primes, factors should be large
            # But they could still be <= p
            if factor <= p and (n // factor) <= p:
                k = i
                continue
            # Check more carefully
            def is_p_smooth_single(n, p):
                if n <= 1:
                    return True
                if n <= p:
                    return True
                if is_prime_mr(n):
                    return n <= p
                f = pollard_brent(n)
                return is_p_smooth_single(f, p) and is_p_smooth_single(n // f, p)
            
            if is_p_smooth_single(n, p):
                k = i
            else:
                break
        return k

    # Search for best p
    best_k = 0
    best_p = 2

    # Phase 1: Test small primes using segmented sieve
    # For small p, the segmented sieve is very fast
    for p in small_primes:
        if time.time() - start_time > DEADLINE:
            break
        k = check_run_segmented(p, 200)
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 200:
                break

    # Phase 2: Test larger primes
    if best_k < 200 and time.time() - start_time < DEADLINE:
        # Generate larger primes and test them
        # Use a simple prime generator for larger ranges
        def next_prime(n):
            if n <= 2:
                return 2
            if n % 2 == 0:
                n += 1
            else:
                n += 2
            while not is_prime_mr(n):
                n += 2
            return n

        # Test primes in various ranges
        # Focus on ranges where long runs might occur
        test_start = max(small_primes[-1] + 2, 1000000)
        
        # Sample primes in larger ranges
        ranges_to_test = [
            (10**6, 5 * 10**6),
            (5 * 10**6, 2 * 10**7),
            (2 * 10**7, 10**8),
            (10**8, 5 * 10**8),
            (5 * 10**8, 10**9),
        ]
        
        for lo, hi in ranges_to_test:
            if time.time() - start_time > DEADLINE:
                break
            # Sample primes in this range
            # Use random sampling to find primes
            n_samples = max(100, int((hi - lo) / 100000))
            for _ in range(n_samples):
                if time.time() - start_time > DEADLINE:
                    break
                # Pick a random odd number in range
                candidate = random.randint(lo, hi)
                if candidate % 2 == 0:
                    candidate += 1
                # Find next prime
                p = next_prime(candidate)
                if p > hi:
                    continue
                k = check_run_segmented(p, 200)
                if k > best_k:
                    best_k = k
                    best_p = p
                    if best_k >= 200:
                        break
            if best_k >= 200:
                break

    # Final fallback: if we haven't found a good k, use a known good result
    # p=2 gives k=1 (2^2+1=5, which is prime > 2, so k=0... wait)
    # Actually for p=2: 4+1=5 (prime>2), so k=0
    # For p=3: 9+1=10=2*5 (5>3), so k=0
    # For p=5: 25+1=26=2*13 (13>5), so k=0
    # For p=7: 49+1=50=2*5*5 (all <=7), 49+2=51=3*17 (17>7), so k=1
    # Let's verify our search found at least something
    
    if best_k == 0:
        # Fallback: try a few more primes
        for p in [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
            if time.time() - start_time > DEADLINE:
                break
            k = check_run_segmented(p, 200)
            if k > best_k:
                best_k = k
                best_p = p

    # Ensure we have a valid result
    if best_k < 1:
        # Find at least one valid p with k >= 1
        # Try small primes more carefully
        for p in small_primes[:100]:
            if time.time() - start_time > DEADLINE:
                break
            k = check_run_segmented(p, 200)
            if k >= 1:
                best_k = k
                best_p = p
                break
        if best_k < 1:
            # Absolute fallback
            best_p = 7
            best_k = 1  # 49+1=50=2*5*5, all factors <= 7

    return {'p': best_p, 'k': best_k}
