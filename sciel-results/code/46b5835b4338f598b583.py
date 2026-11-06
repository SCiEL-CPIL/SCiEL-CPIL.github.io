"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    deadline = time.time() + 115.0  # safety margin before 150s kill

    # Precompute small primes up to 10^5 for trial division
    def sieve_primes(n):
        if n < 2:
            return []
        bs = bytearray(b'\x01') * (n + 1)
        bs[0] = bs[1] = 0
        for i in range(2, int(math.isqrt(n)) + 1):
            if bs[i]:
                bs[i*i::i] = bytearray(len(bs[i*i::i]))
        return [i for i in range(2, n + 1) if bs[i]]

    SMALL_LIMIT = 10**5
    small_primes = sieve_primes(SMALL_LIMIT)

    # Deterministic Miller-Rabin for n < 3.2e18
    def is_prime(n):
        if n < 2:
            return False
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return n == p
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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

    # Pollard's rho with Brent's improvement
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
            c = random.randrange(1, n - 1)
            y = random.randrange(0, n - 1)
            m = 128
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
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
            if g != n:
                return g

    def is_p_smooth(n, p):
        """Check if all prime factors of n are <= p"""
        if n <= 1:
            return True
        if is_prime(n):
            return n <= p
        # Trial division by small primes
        for sp in small_primes:
            if sp * sp > n:
                break
            if n % sp == 0:
                if sp > p:
                    return False
                n //= sp
                if not is_p_smooth(n, p):
                    return False
                return True
        if n == 1:
            return True
        if is_prime(n):
            return n <= p
        d = pollard_brent(n)
        if d == n:
            return n <= p
        if not is_p_smooth(d, p):
            return False
        if not is_p_smooth(n // d, p):
            return False
        return True

    def fast_residual_check(base, p, max_k=200):
        """
        For numbers base+1 .. base+max_k, divide out all small primes <= min(p, SMALL_LIMIT).
        Return a list of (i, residual) where residual > 1.
        If any residual > p, the number is definitely not p-smooth.
        """
        # Determine which small primes to use: those <= p and <= SMALL_LIMIT
        # Since p can be up to 10^9, we use all small_primes (up to 10^5)
        # but only those <= p
        limit = min(p, SMALL_LIMIT)
        # Find the index up to which small_primes[i] <= limit
        # Binary search
        lo, hi = 0, len(small_primes)
        while lo < hi:
            mid = (lo + hi) // 2
            if small_primes[mid] <= limit:
                lo = mid + 1
            else:
                hi = mid
        num_small = lo  # small_primes[0:num_small] are all <= limit

        residuals = []
        for i in range(1, max_k + 1):
            n = base + i
            # Divide out small primes
            for sp in small_primes[:num_small]:
                if sp * sp > n:
                    break
                if n % sp == 0:
                    while n % sp == 0:
                        n //= sp
            if n > 1:
                residuals.append((i, n))
        return residuals

    def check_window(base, p, max_k=200):
        """
        Check if all numbers base+1 .. base+max_k are p-smooth.
        Returns the run length k (0 to max_k).
        Uses fast residual check first, then full factorization only for survivors.
        """
        # Fast check: get residuals after dividing small primes
        residuals = fast_residual_check(base, p, max_k)
        
        # Build a set of indices that have residual > 1
        # For each residual, check if it's p-smooth
        # If any is not, we need to find where the run breaks
        
        # Strategy: process residuals in order, find the first one that fails
        # Numbers without residuals are automatically p-smooth (all factors were small)
        
        # Create a boolean array: is_p_smooth[i] for i in 1..max_k
        # Initially assume all are smooth, then check residuals
        # A number is smooth if its residual is 1 or all factors of residual are <= p
        
        # For efficiency, check residuals in order of i
        # The run breaks at the first i where the number is not p-smooth
        
        # Group residuals by their index
        # Check each residual: if residual <= p, it's smooth (prime or 1)
        # If residual > p, need to check if it's p-smooth via factorization
        
        # To find the run length, we need the minimum i where number is NOT smooth
        # Numbers with no residual are smooth.
        # Numbers with residual r: smooth iff r is p-smooth
        
        # Check residuals in increasing order of i
        # But we need to find the first failure
        # Sort residuals by i
        residuals.sort(key=lambda x: x[0])
        
        for i, r in residuals:
            if r <= p:
                # r is either 1 (smooth) or a prime <= p (smooth)
                # Actually r could be composite with factors > p? No, r > 1 and r <= p
                # r could be composite like 4, 6, etc. But all its prime factors are <= r <= p
                # So r is p-smooth
                continue
            else:
                # r > p, need to check if r is p-smooth
                # r has no prime factors <= SMALL_LIMIT (we divided those out)
                # So r is either prime or product of primes > SMALL_LIMIT
                # Check if r is prime
                if is_prime(r):
                    # r is prime > p, so NOT p-smooth
                    return i - 1
                else:
                    # r is composite, all prime factors > SMALL_LIMIT > p? 
                    # Wait, SMALL_LIMIT = 10^5, p could be up to 10^9
                    # If p >= SMALL_LIMIT, then we divided out all primes <= p
                    # So r has no prime factors <= p, meaning r is NOT p-smooth (if r > 1)
                    # If p < SMALL_LIMIT, we only divided primes <= p, so r could have factors in (p, SMALL_LIMIT]
                    # But we divided all primes <= min(p, SMALL_LIMIT)
                    # So r has no prime factors <= min(p, SMALL_LIMIT)
                    # If p < SMALL_LIMIT: r has no factors <= p, so r is NOT p-smooth
                    # If p >= SMALL_LIMIT: r has no factors <= SMALL_LIMIT <= p, so r is NOT p-smooth
                    # In both cases, r > 1 and r has no prime factors <= p, so r is NOT p-smooth
                    return i - 1
        
        # All numbers are p-smooth
        return max_k

    def find_smooth_run(p, max_k=200):
        """Find the longest run of p-smooth numbers starting at p^2+1"""
        base = p * p
        return check_window(base, p, max_k)

    # Generate candidate primes and test them
    best_k = 0
    best_p = 2

    # 1. Test small primes first (fast factorization)
    for p in small_primes:
        if time.time() > deadline - 5:
            break
        k = find_smooth_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    # 2. Search larger primes using segmented sieve
    def segmented_sieve(low, high):
        """Generate primes in [low, high]"""
        if high < 2:
            return []
        limit = int(math.isqrt(high)) + 1
        if limit <= len(small_primes) and small_primes[-1] >= limit:
            small = [p for p in small_primes if p <= limit]
        else:
            small = sieve_primes(limit)
        sieve = bytearray(b'\x01') * (high - low + 1)
        for p in small:
            if p * p > high:
                break
            start = max(p * p, ((low + p - 1) // p) * p)
            for j in range(start, high + 1, p):
                sieve[j - low] = 0
        primes = []
        for i in range(low, high + 1):
            if sieve[i - low] and i >= 2:
                primes.append(i)
        return primes

    # Search ranges
    current_low = 10**5
    current_high = 10**6

    while current_high <= 10**9 and time.time() < deadline - 10:
        candidates = segmented_sieve(current_low, current_high)
        
        for p in candidates:
            if time.time() > deadline - 5:
                break
            
            # Quick pre-check: test first 3 numbers using fast residual
            base = p * p
            # Check if base+1 is p-smooth
            n1 = base + 1
            # Divide small primes
            temp = n1
            limit = min(p, SMALL_LIMIT)
            for sp in small_primes:
                if sp > limit or sp * sp > temp:
                    break
                if temp % sp == 0:
                    while temp % sp == 0:
                        temp //= sp
            if temp > 1 and temp > p:
                continue  # not p-smooth
            
            n2 = base + 2
            temp = n2
            for sp in small_primes:
                if sp > limit or sp * sp > temp:
                    break
                if temp % sp == 0:
                    while temp % sp == 0:
                        temp //= sp
            if temp > 1 and temp > p:
                continue  # not p-smooth
            
            n3 = base + 3
            temp = n3
            for sp in small_primes:
                if sp > limit or sp * sp > temp:
                    break
                if temp % sp == 0:
                    while temp % sp == 0:
                        temp //= sp
            if temp > 1 and temp > p:
                continue  # not p-smooth
            
            # If pre-check passes, do full check
            k = find_smooth_run(p)
            if k > best_k:
                best_k = k
                best_p = p
        
        # Increase range
        current_low = current_high
        current_high = current_high * 10

    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
