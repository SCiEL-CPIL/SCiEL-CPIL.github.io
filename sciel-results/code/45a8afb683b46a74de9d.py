import math
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # safety margin for 120s limit
    
    # Deterministic Miller-Rabin for n < 2^64
    def is_prime_mr(n):
        if n < 2:
            return False
        if n == 2 or n == 3:
            return True
        if n % 2 == 0:
            return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        # Bases sufficient for n < 2^64
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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
        while True:
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
            m = random.randint(1, n - 1)
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
            # retry

    def factorize(n):
        """Fully factor n, returning list of prime factors with multiplicity."""
        if n <= 1:
            return []
        if n < 4:
            return [n]
        # Small primes trial division
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        factors = []
        for p in small_primes:
            while n % p == 0:
                factors.append(p)
                n //= p
        if n == 1:
            return factors
        if is_prime_mr(n):
            factors.append(n)
            return factors
        # Pollard-Brent
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime_mr(m):
                factors.append(m)
            else:
                d = pollard_brent(m)
                if d == m:
                    # retry
                    stack.append(m)
                else:
                    stack.append(d)
                    stack.append(m // d)
        return factors

    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p."""
        if n <= 1:
            return True
        factors = factorize(n)
        for f in factors:
            if f > p:
                return False
        return True

    def check_run(p, max_k=200):
        """Check how many consecutive integers p^2+1, ..., p^2+k are p-smooth."""
        p2 = p * p
        k = 0
        for i in range(1, max_k + 1):
            n = p2 + i
            if not is_p_smooth(n, p):
                break
            k = i
        return k

    # Sieve for primes up to limit
    def sieve_primes(limit):
        if limit < 2:
            return []
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    # Strategy: 
    # 1. Search small primes (up to ~50000) with full check
    # 2. Then use a "smoothness sieve" approach for larger primes:
    #    For a given p, the condition is that p^2+1, ..., p^2+k are all p-smooth.
    #    Key insight: if p^2 + i has a prime factor > p, it fails.
    #    For large p, p^2+i is ~p^2, so any prime factor > p must be in (p, p^2].
    #    The number p^2+i is p-smooth iff all its prime factors are <= p.
    #    
    #    A faster check: for each i in 1..k, check if p^2+i has any prime factor > p.
    #    Since p^2+i < p^2 + 200, and p is large, p^2+i is either prime or has small factors.
    #    If p^2+i is prime, it's > p (for i>=1), so it fails.
    #    So first check: is p^2+i composite with all factors <= p?
    #    
    #    Better: use the fact that we can quickly check if n is p-smooth by:
    #    1. Trial divide by small primes
    #    2. Check if remainder is 1 or a prime <= p
    
    # For efficiency, let's use a different approach for large p:
    # Precompute small primes for trial division
    small_primes = sieve_primes(100000)
    
    best_k = 0
    best_p = 2
    
    # Phase 1: Check small primes up to 100000
    primes_small = sieve_primes(100000)
    for p in primes_small:
        if time.time() > deadline - 30:
            break
        k = check_run(p, 200)
        if k > best_k:
            best_k = k
            best_p = p
            # If we found a good one, continue searching for better
    
    # Phase 2: For larger primes, use a smarter approach
    # For a prime p, p^2+1 is p-smooth iff it's not prime (since if prime, it's > p)
    # and all its prime factors are <= p.
    # 
    # Key optimization: p^2 + i for i in 1..k. If any of these is prime, it fails (since prime > p).
    # So we need p^2+1, ..., p^2+k all composite AND all prime factors <= p.
    #
    # For large p, p^2+i is likely to have small prime factors.
    # The critical check is whether all prime factors are <= p.
    # Since p^2+i < p^2+p for i < p, any prime factor > p would make the number 
    # either prime or have a factor in (p, p^2+i].
    #
    # Actually, a number n = p^2+i where i < p: if n has a prime factor q > p, then 
    # n/q < (p^2+p)/p = p+1, so n/q < p+1. If n/q is also > 1, then n/q is a factor < p+1.
    # So n = q * m where q > p and m < p+1. This means n is divisible by some m < p+1 
    # and n/m > p.
    #
    # So p^2+i is NOT p-smooth iff there exists a divisor d of (p^2+i) with 1 < d <= p 
    # such that (p^2+i)/d > p, OR (p^2+i) is prime.
    # Equivalently: p^2+i is p-smooth iff for every divisor d of (p^2+i) with d <= p, 
    # we have (p^2+i)/d <= p, i.e., d >= (p^2+i)/p ≈ p.
    # This is getting complex. Let's just use the factorization approach but with 
    # early termination.
    
    # Phase 2: Random sampling of larger primes
    # Use congruence to generate candidate primes
    random.seed(42)
    
    # For large p, the key constraint is that p^2+1 through p^2+k are all p-smooth.
    # A necessary condition: p^2+i must be composite for all i in 1..k (since if prime, > p).
    # Also, p^2+i must not have any prime factor in (p, p^2+i].
    
    # Let's try primes in ranges where p^2+i might be smooth.
    # For a fixed small prime q, p^2+i ≡ 0 (mod q) means p^2 ≡ -i (mod q).
    # This has solutions if -i is a quadratic residue mod q.
    # So for each i, the set of p (mod q) for which q | (p^2+i) is determined.
    
    # Strategy: For a target k, we want p such that for all i=1..k, p^2+i is p-smooth.
    # This is a strong condition. Let's search more cleverly.
    
    # Try medium-sized primes (100000 to 10^6) with a faster check
    def check_run_fast(p, max_k=200):
        """Faster check using trial division with early exit."""
        p2 = p * p
        k = 0
        for i in range(1, max_k + 1):
            n = p2 + i
            r = n
            # Trial divide by small primes
            for q in small_primes:
                if q * q > r:
                    break
                if q > p:
                    break
                while r % q == 0:
                    r //= q
            if r == 1:
                k = i
                continue
            # r > 1 after trial division
            if is_prime_mr(r):
                if r <= p:
                    k = i
                    continue
                else:
                    break
            else:
                # r is composite, need to check if all factors <= p
                # Factor r
                factors = factorize(r)
                if all(f <= p for f in factors):
                    k = i
                else:
                    break
        return k

    # Search medium primes
    primes_medium = sieve_primes(1000000)
    for p in primes_medium[10000:]:  # skip small ones already checked
        if time.time() > deadline - 15:
            break
        k = check_run_fast(p, 200)
        if k > best_k:
            best_k = k
            best_p = p
    
    # Phase 3: Try some larger primes with the fast check
    if time.time() < deadline - 10:
        # Try primes around 10^6 to 10^7
        # Generate primes in that range using segmented sieve
        def segmented_sieve(low, high):
            """Generate primes in [low, high]."""
            limit = int(math.isqrt(high)) + 1
            small_primes_seg = sieve_primes(limit)
            size = high - low + 1
            is_prime_seg = bytearray(b'\x01') * size
            for p in small_primes_seg:
                start = ((low + p - 1) // p) * p
                if start == p:
                    start += p
                for j in range(start - low, size, p):
                    is_prime_seg[j] = 0
            primes = []
            for i in range(size):
                if is_prime_seg[i]:
                    n = low + i
                    if n >= 2:
                        primes.append(n)
            return primes

        # Try a few ranges
        ranges = [(1000000, 2000000), (2000000, 3000000), (3000000, 4000000)]
        for low, high in ranges:
            if time.time() > deadline - 5:
                break
            seg_primes = segmented_sieve(low, high)
            for p in seg_primes:
                if time.time() > deadline - 3:
                    break
                k = check_run_fast(p, 200)
                if k > best_k:
                    best_k = k
                    best_p = p
    
    return {'p': best_p, 'k': best_k}
