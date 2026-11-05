"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import random

def construct():
    deadline = time.time() + 115.0  # safety margin before 120s limit
    
    # Deterministic Miller-Rabin for 64-bit integers
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(s - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True
    
    # Pollard's Rho with Brent's cycle detection
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
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
                while True:
                    ys = (ys * ys + c) % n
                    g = math.gcd(abs(x - ys), n)
                    if g > 1:
                        break
            if g != n:
                return g
    
    # Check if all prime factors of v are <= limit
    def all_factors_leq(v, limit):
        if v <= 1:
            return True
        if is_prime(v):
            return v <= limit
        n = v
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for p in small_primes:
            while n % p == 0:
                if p > limit:
                    return False
                n //= p
        if n == 1:
            return True
        if is_prime(n):
            return n <= limit
        stack = [n]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime(m):
                if m > limit:
                    return False
            else:
                d = pollard_brent(m)
                if d == m:
                    stack.append(m)
                else:
                    stack.append(d)
                    stack.append(m // d)
        return True
    
    # Segmented sieve for primes in [lo, hi]
    def segmented_sieve(lo, hi, small_primes):
        if hi < 2:
            return []
        size = hi - lo + 1
        seg = bytearray(b'\x01') * size
        if lo == 1:
            seg[0] = 0
        for p in small_primes:
            if p * p > hi:
                break
            start = max(p * p, ((lo + p - 1) // p) * p)
            for j in range(start, hi + 1, p):
                seg[j - lo] = 0
        return [lo + i for i in range(size) if seg[i]]
    
    # Fast p-smoothness check for a contiguous block using segmented sieve
    # Returns the longest run length starting at offset 0 (i.e., p^2+1, p^2+2, ...)
    def max_p_smooth_run(p, max_k=200):
        """Check p^2+1, p^2+2, ..., p^2+max_k for p-smoothness.
        Returns the length of the longest consecutive run starting from p^2+1."""
        base = p * p
        lo = base + 1
        hi = base + max_k
        
        # For small p, use direct factorization
        if p < 1000:
            k = 0
            for i in range(1, max_k + 1):
                if time.time() > deadline:
                    break
                if all_factors_leq(base + i, p):
                    k += 1
                else:
                    break
            return k
        
        # For larger p, use segmented sieve to find all prime factors in [lo, hi]
        # We need primes up to sqrt(hi) for trial division, but for p-smoothness
        # we only care about factors > p
        # 
        # Strategy: For each number in [lo, hi], we need to check if it has any
        # prime factor > p. We can do this by:
        # 1. Divide out all prime factors <= p
        # 2. Check if remainder is 1 or prime <= p
        #
        # But a faster approach: use a segmented sieve to find which numbers in [lo,hi]
        # are NOT p-smooth. A number is not p-smooth if it has a prime factor > p.
        # 
        # Alternative: For each number n in [lo, hi], compute n and check if after
        # removing all factors <= p, the remainder is 1.
        #
        # Efficient approach: Precompute all primes <= p using a sieve, then for each
        # number in the range, divide out those primes.
        
        # Get all primes <= p for trial division
        # For p up to ~10^6, this is feasible
        if p <= 500000:
            # Sieve primes up to p
            is_p = bytearray(b'\x01') * (p + 1)
            is_p[0] = is_p[1] = 0
            for i in range(2, int(math.isqrt(p)) + 1):
                if is_p[i]:
                    step = i
                    start = i * i
                    is_p[start:p+1:step] = bytearray(len(range(start, p+1, step)))
            primes_leq_p = [i for i in range(2, p + 1) if is_p[i]]
            
            # For each number in [lo, hi], check p-smoothness
            k = 0
            for i in range(1, max_k + 1):
                if time.time() > deadline:
                    break
                n = base + i
                # Divide out all prime factors <= p
                for q in primes_leq_p:
                    if q * q > n:
                        break
                    while n % q == 0:
                        n //= q
                # After dividing out all primes <= p, n should be 1 or a prime <= p
                if n > 1:
                    if n <= p:
                        # n is a prime (or 1) <= p, so it's fine
                        # Actually n could be composite with all factors <= p, but we divided those out
                        # So n must be 1 or prime > p... wait, we divided out all primes <= p
                        # So if n > 1, it must be prime and > p (since all its prime factors > p)
                        # OR it could be that n is prime <= p but we didn't divide it because q*q > n
                        # Let's check: if n is prime and <= p, then it's fine
                        if is_prime(n) and n <= p:
                            pass  # OK
                        else:
                            # n has a prime factor > p
                            break
                    else:
                        # n > p, and after removing all prime factors <= p, n > 1
                        # This means n has a prime factor > p
                        break
                k += 1
            return k
        else:
            # For very large p, use the slower method
            k = 0
            for i in range(1, max_k + 1):
                if time.time() > deadline:
                    break
                if all_factors_leq(base + i, p):
                    k += 1
                else:
                    break
            return k
    
    best_k = 0
    best_p = 2
    
    # Check p = 2
    if time.time() < deadline:
        k = 0
        while time.time() < deadline:
            v = 4 + k + 1
            if all_factors_leq(v, 2):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = 2
    
    # Search primes in increasing ranges
    # First, small primes up to 10^6
    limit = 10**6
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(math.isqrt(limit)) + 1):
        if is_p[i]:
            step = i
            start = i * i
            is_p[start:limit+1:step] = bytearray(len(range(start, limit+1, step)))
    primes = [i for i in range(2, limit+1) if is_p[i]]
    
    for p in primes:
        if time.time() >= deadline:
            break
        if p == 2:
            continue
        k = max_p_smooth_run(p, 200)
        if k > best_k:
            best_k = k
            best_p = p
    
    # If we still have time, search larger primes using segmented sieve
    if time.time() < deadline - 20:
        # Search in segments up to 10^7
        seg_limit = 10**7
        seg_size = 10**6
        # Small primes for segmented sieve (primes up to sqrt(seg_limit))
        small_lim = int(math.isqrt(seg_limit)) + 1
        is_sp = bytearray(b'\x01') * (small_lim + 1)
        is_sp[0] = is_sp[1] = 0
        for i in range(2, int(math.isqrt(small_lim)) + 1):
            if is_sp[i]:
                step = i
                start = i * i
                is_sp[start:small_lim+1:step] = bytearray(len(range(start, small_lim+1, step)))
        small_primes = [i for i in range(2, small_lim+1) if is_sp[i]]
        
        start = limit + 1
        while start <= seg_limit and time.time() < deadline - 10:
            end = min(start + seg_size - 1, seg_limit)
            seg_primes = segmented_sieve(start, end, small_primes)
            for p in seg_primes:
                if time.time() >= deadline - 5:
                    break
                k = max_p_smooth_run(p, 200)
                if k > best_k:
                    best_k = k
                    best_p = p
            start = end + 1
    
    # If we still have time, search even larger primes
    if time.time() < deadline - 30:
        seg_limit2 = min(10**8, 10**9)
        seg_size2 = 10**7
        start2 = seg_limit + 1
        while start2 <= seg_limit2 and time.time() < deadline - 15:
            end2 = min(start2 + seg_size2 - 1, seg_limit2)
            seg_primes2 = segmented_sieve(start2, end2, small_primes)
            for p in seg_primes2:
                if time.time() >= deadline - 5:
                    break
                k = max_p_smooth_run(p, 200)
                if k > best_k:
                    best_k = k
                    best_p = p
            start2 = end2 + 1
    
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
