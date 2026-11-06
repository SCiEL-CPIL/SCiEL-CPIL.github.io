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

    # Segmented sieve to check p-smoothness of range [p^2+1, p^2+k]
    # Returns the run length: longest prefix where all numbers are p-smooth
    def check_smooth_run_batch(p, max_k=200):
        """
        Use segmented sieve to find the longest run of p-smooth numbers
        in [p^2+1, p^2+max_k].
        
        Strategy:
        1. Generate all primes <= p (or up to sqrt(p^2+max_k))
        2. For each such prime q, mark multiples in the range
        3. After sieving, each position has its remaining factor
        4. If remaining factor is 1 or a prime <= p, it's p-smooth
        """
        p_sq = p * p
        lo = p_sq + 1
        hi = p_sq + max_k
        size = hi - lo + 1  # max_k
        
        # We need primes up to sqrt(hi) for complete factorization
        # But for p-smoothness, we only need to divide by primes <= p
        # If after dividing by all primes <= p, the remainder is 1 or prime <= p, it's smooth
        
        # Generate primes up to min(p, sqrt(hi))
        # sqrt(p^2 + 200) ≈ p for large p, so we need primes up to ~p
        # But p can be up to 10^9, so we can't sieve up to p
        
        # Alternative approach: 
        # - Sieve with small primes (up to some limit L)
        # - For remaining composite parts, check if they have factors > p
        
        # Better: use trial division with primes up to min(p, 10^6) first,
        # then for the remaining part, check if it's 1, prime <= p, or composite with all factors <= p
        
        # Let's use a two-phase approach:
        # Phase 1: Divide by all primes <= L (where L is manageable, say 10^6)
        # Phase 2: For each number's remaining factor, check p-smoothness
        
        # Generate primes up to L
        L = min(p, 10**6)
        if L < 2:
            L = 2
        
        # Sieve primes up to L
        if L <= 10**7:
            small_primes = sieve(L)
        else:
            # Fallback: use primes up to 10^7
            small_primes = sieve(10**7)
            small_primes = [q for q in small_primes if q <= p]
        
        # Initialize remaining values
        rem = list(range(lo, hi + 1))
        
        # Sieve: divide out all small prime factors
        for q in small_primes:
            if q > p:
                break
            # Find first multiple of q in [lo, hi]
            start = ((lo + q - 1) // q) * q
            if start < lo:
                start += q
            for pos in range(start - lo, size, q):
                while rem[pos] % q == 0:
                    rem[pos] //= q
        
        # Now check each position: is rem[i] p-smooth?
        # rem[i] should be 1, or a prime <= p, or a product of primes all <= p
        # Since we've divided by all primes <= min(p, 10^6), 
        # if p <= 10^6, rem[i] is either 1 or prime
        # if p > 10^6, rem[i] could be composite with factors > 10^6 but <= p
        
        run = 0
        for i in range(size):
            r = rem[i]
            if r == 1:
                run += 1
                continue
            # Check if r is p-smooth
            # If p <= 10^6: r must be prime and r <= p (since we divided all primes <= p)
            # If p > 10^6: r could be composite; need to verify all factors <= p
            
            if p <= 10**6:
                # r should be 1 or a prime > 10^6... but we divided all primes <= p
                # Actually if p <= 10^6, we divided all primes <= p, so r is either 1 or has no prime factor <= p
                # So r is p-smooth iff r == 1 or (r is prime and r <= p)
                # But wait, if r is composite, it has a prime factor <= sqrt(r) <= sqrt(p^2+200) ≈ p
                # Hmm, this needs more care.
                # Actually: if we divided by ALL primes <= p, and p <= 10^6, then r has no prime factor <= p.
                # So r is p-smooth iff r == 1.
                # Wait no: r could be a prime > p, or 1, or composite with all factors > p.
                # For r to be p-smooth, all prime factors of r must be <= p.
                # Since we removed all prime factors <= p, r has no prime factors <= p.
                # So r is p-smooth iff r == 1.
                if r == 1:
                    run += 1
                else:
                    break
            else:
                # p > 10^6: r has no prime factors <= 10^6
                # r could be: 1, prime > 10^6, or composite with factors > 10^6
                # For r to be p-smooth, all prime factors must be <= p
                # Check if r is prime:
                if is_prime(r):
                    if r <= p:
                        run += 1
                    else:
                        break
                else:
                    # r is composite with no factors <= 10^6
                    # Its smallest prime factor is > 10^6
                    # For r to be p-smooth, we need all factors <= p
                    # Since smallest factor > 10^6, we need 10^6 < smallest_factor <= p
                    # Let's factor r and check
                    factors = factorize(r)
                    if all(f <= p for f in factors):
                        run += 1
                    else:
                        break
        
        return run

    # Generate candidate primes
    limit = 5000000
    primes = sieve(limit)
    
    best_k = 0
    best_p = 2
    
    # Use batch smoothness check for all primes
    for p in primes:
        if time.time() > deadline:
            break
        
        k = check_smooth_run_batch(p, 200)
        if k > best_k:
            best_k = k
            best_p = p
    
    # If we have time, try some larger primes (up to 10^9)
    if time.time() < deadline - 10:
        # Try primes in a higher range
        limit2 = 10**7
        primes2 = sieve(limit2)
        for p in primes2:
            if time.time() > deadline:
                break
            if p <= limit:
                continue
            k = check_smooth_run_batch(p, 200)
            if k > best_k:
                best_k = k
                best_p = p
    
    # If still have time, try random primes up to 10^9
    if time.time() < deadline - 5:
        random.seed(42)
        for _ in range(200):
            if time.time() > deadline:
                break
            candidate = random.randint(10**7, 10**9)
            while not is_prime(candidate):
                candidate += 1
            if candidate > 10**9:
                continue
            if candidate < limit2:
                continue
            k = check_smooth_run_batch(candidate, 200)
            if k > best_k:
                best_k = k
                best_p = candidate
    
    return {'p': best_p, 'k': best_k}
