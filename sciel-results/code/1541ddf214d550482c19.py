"""New program: Segmented sieve for larger primes, optimized smoothness check."""
# EVOLVE-BLOCK-START

def construct():
    import math
    import random

    # ---------- Deterministic Miller-Rabin for 64-bit ----------
    def is_prime(n):
        if n < 2:
            return False
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for p in small_primes:
            if n == p:
                return True
            if n % p == 0:
                return False
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
                x = (x * x) % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    # ---------- Pollard's Rho (Brent) ----------
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(1, n - 1)
            c = random.randint(1, n - 1)
            m = 128
            g = r = q = 1
            x = ys = y
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
                    g = gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = gcd(abs(x - ys), n)
            if g != n:
                return g

    def has_prime_factor_greater_than(n, limit):
        """
        Returns True if n has any prime factor > limit.
        """
        if n == 1:
            return False
        if n <= limit:
            return False # All factors <= n <= limit
        
        # Trial division by small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        temp = n
        for p in small_primes:
            while temp % p == 0:
                temp //= p
        if temp == 1:
            return False
        
        # Check if temp is prime
        if is_prime(temp):
            return temp > limit
        
        # Factor temp using Pollard-Brent
        factors = []
        def factor_recursive(x):
            if x == 1:
                return
            if is_prime(x):
                factors.append(x)
                return
            d = pollard_brent(x)
            factor_recursive(d)
            factor_recursive(x // d)
            
        factor_recursive(temp)
        
        for f in factors:
            if f > limit:
                return True
        return False

    # ---------- Segmented Sieve for primes in [low, high] ----------
    def segmented_sieve(low, high):
        size = high - low + 1
        sieve = [True] * size
        limit = int(math.isqrt(high))
        
        # Generate small primes up to sqrt(high)
        small_primes = []
        is_small_prime = [True] * (limit + 1)
        for i in range(2, limit + 1):
            if is_small_prime[i]:
                small_primes.append(i)
                for j in range(i*i, limit + 1, i):
                    is_small_prime[j] = False
            
        for p in small_primes:
            # Find the first multiple of p in [low, high]
            start = ((low + p - 1) // p) * p
            if start == p:
                start += p
            for j in range(start, high + 1, p):
                sieve[j - low] = False
                
        primes = []
        for i in range(low, high + 1):
            if sieve[i - low]:
                primes.append(i)
        return primes

    # ---------- Search ----------
    best_k = 0
    best_p = 2
    
    # Search in segments to cover a wide range efficiently
    # We'll check segments of size 100,000
    # Start from 100,000 up to 10,000,000 (10M)
    # This is a different range than the current program (which did 0-2M)
    # Actually, let's go higher. 1M to 10M.
    
    current_time = 0.0
    import time
    start_time = time.time()
    TIME_LIMIT = 110.0 # Leave 10s margin
    
    # We will iterate through segments
    # Segment size 100,000
    SEG_SIZE = 100_000
    start_range = 1_000_000
    end_range = 10_000_000
    
    for low in range(start_range, end_range, SEG_SIZE):
        high = min(low + SEG_SIZE - 1, end_range)
        if time.time() - start_time > TIME_LIMIT:
            break
        
        primes = segmented_sieve(low, high)
        
        for p in primes:
            if time.time() - start_time > TIME_LIMIT:
                break
            
            p2 = p * p
            k = 0
            for offset in range(1, 201):
                val = p2 + offset
                # Check if val has a prime factor > p
                if has_prime_factor_greater_than(val, p):
                    break
                k = offset
            
            if k > best_k:
                best_k = k
                best_p = p
                
    # If we didn't find anything good, fall back to a quick check of small primes
    # Or just return what we have
    if best_k == 0:
        # Fallback to a known small prime or just 2
        best_p = 2
        best_k = 1 # 2^2+1=5, 5>2. So k=0? 
        # Wait, if k=0, is it valid? "1 <= k <= 200".
        # The task says "1 <= k".
        # If no k >= 1 works, we must return a valid k?
        # "every integer ... must have ...".
        # If k=1 fails, we can't return k=1.
        # The current program returns best_k. If best_k is 0, it returns 0?
        # But spec says 1 <= k.
        # Let's ensure we find at least k=1.
        # p=2: 4+1=5 (5>2). Fail.
        # p=3: 9+1=10 (5>3). Fail.
        # p=5: 25+1=26 (13>5). Fail.
        # p=7: 49+1=50 (5<=7, 2<=7). 50=2*5*5. All <=7. OK.
        # 49+2=51 (17>7). Fail.
        # So for p=7, k=1.
        
        # Let's hardcode a check for small primes if the search fails
        for p in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]:
            p2 = p * p
            k = 0
            for offset in range(1, 201):
                val = p2 + offset
                if has_prime_factor_greater_than(val, p):
                    break
                k = offset
            if k > best_k:
                best_k = k
                best_p = p

    return {'p': best_p, 'k': best_k}

# EVOLVE-BLOCK-END
