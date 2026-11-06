import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 110 seconds deadline
    
    # Sieve of Eratosthenes for small primes
    def sieve(limit):
        is_prime = bytearray(b'\x01') * (limit + 1)
        is_prime[0] = is_prime[1] = 0
        for i in range(2, int(limit**0.5) + 1):
            if is_prime[i]:
                is_prime[i*i:limit+1:i] = bytearray(len(range(i*i, limit+1, i)))
        return [i for i, prime in enumerate(is_prime) if prime]
    
    small_primes = sieve(100000) # Primes up to 10^5 for trial division
    small_primes_set = set(small_primes)
    
    # Miller-Rabin primality test
    def is_prime_mr(n):
        if n < 2:
            return False
        # Small primes
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == p:
                return True
            if n % p == 0:
                return False
        # Write n-1 as d*2^r
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        # Bases sufficient for n < 2^64
        witnesses = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for a in witnesses:
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    # Pollard's Rho (Brent variant)
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randint(1, n - 1)
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
                        q = (q * abs(x - y)) % n
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
        if n == 1:
            return []
        if n in small_primes_set:
            return [n]
        factors = []
        # Trial division by small primes
        for p in small_primes:
            if p * p > n:
                break
            if n % p == 0:
                while n % p == 0:
                    factors.append(p)
                    n //= p
        if n == 1:
            return factors
        if is_prime_mr(n):
            factors.append(n)
            return factors
        # Use Pollard-Brent
        stack = [n]
        while stack:
            x = stack.pop()
            if is_prime_mr(x):
                factors.append(x)
            else:
                d = pollard_brent(x)
                stack.append(d)
                stack.append(x // d)
        return factors

    def largest_prime_factor(n):
        if n == 1:
            return 1
        max_f = 1
        temp_n = n
        for p in small_primes:
            if p * p > temp_n:
                break
            if temp_n % p == 0:
                max_f = max(max_f, p)
                while temp_n % p == 0:
                    temp_n //= p
        if temp_n > 1:
            if is_prime_mr(temp_n):
                max_f = max(max_f, temp_n)
            else:
                f = factorize(temp_n)
                max_f = max(max_f, max(f))
        return max_f

    def check_window(p, p_sq, window_size):
        """
        Use segmented sieve to check if all numbers p^2+1 ... p^2+window_size
        have all prime factors <= p.
        Returns the run length k (0 to window_size).
        """
        # For each number in the window, we need to verify it has no prime factor > p.
        # Strategy: For each prime q > p that could divide some number in the window,
        # mark those numbers. If any number gets marked, it has a factor > p.
        # 
        # A prime q > p can divide p^2 + j iff p^2 + j ≡ 0 (mod q) iff j ≡ -p^2 (mod q).
        # Since q > p, q^2 > p^2. For q to divide p^2+j where 1 <= j <= window_size,
        # we need q <= p^2 + window_size.
        # Also q > p.
        #
        # Instead of iterating over all primes q > p (which is too many), 
        # we can check each number p^2+j individually by factoring it,
        # but use a fast pre-check: if p^2+j is prime, it's > p (since p^2+j > p for p>=2),
        # so it fails immediately.
        #
        # Better approach: For each j in 1..window_size, check if p^2+j has all factors <= p.
        # We do this by trial dividing by primes <= p, and if the remainder is > 1, it's a factor > p.
        # But trial division by all primes <= p is too slow for large p.
        #
        # Key insight: p^2 + j is at most p^2 + 200. Its prime factors are either <= p or > p.
        # If we divide out all prime factors <= p (using small primes up to some limit),
        # the remaining factor must be <= p for the number to pass.
        # 
        # Actually, a simpler fast check: 
        # p^2 + j has a prime factor > p iff it is NOT p-smooth.
        # We can check by trying to divide out small primes and seeing if remainder is 1 or prime <= p.
        #
        # Let's use a hybrid: trial divide by primes up to min(p, some_limit), 
        # then if remainder > 1, check if it's prime and <= p.
        
        max_trial = min(p, 100000)  # Trial divide by primes up to min(p, 100000)
        trial_primes = [q for q in small_primes if q <= max_trial]
        
        k = 0
        for j in range(1, window_size + 1):
            n = p_sq + j
            # Quick check: if n is prime, then its only factor is itself > p (for p >= 2), so fail
            # But checking primality is expensive. Let's just do trial division.
            
            temp = n
            for q in trial_primes:
                if q * q > temp:
                    break
                while temp % q == 0:
                    temp //= q
            
            if temp == 1:
                k = j
            else:
                # temp > 1: either temp is prime or composite
                # If temp <= p, it's fine (all factors <= p)
                # If temp > p, it has a factor > p, so fail
                if temp <= p:
                    # Need to verify temp's factors are all <= p
                    # Since we've divided by all primes <= max_trial, and max_trial >= p if p <= 100000,
                    # temp should be 1 or prime. If p > 100000, temp could be composite with factors > 100000.
                    # But if temp <= p and we've divided by all primes <= min(p, 100000),
                    # then if p <= 100000, temp must be 1 (since we divided by all primes <= p).
                    # If p > 100000, temp could be a prime > 100000 but <= p, which is fine.
                    # Or temp could be composite with all factors > 100000 but <= p.
                    # In any case, if temp <= p, all its prime factors are <= temp <= p. So it's fine.
                    k = j
                else:
                    # temp > p, so it has a prime factor > p. Fail.
                    break
        
        return k

    best_k = 0
    best_p = 2
    
    # Generate primes for p
    limit_p = 10**7
    primes_p = sieve(limit_p)
    
    for p in primes_p:
        if time.time() > deadline:
            break
        
        p_sq = p * p
        # Only need to check up to best_k + 1 to see if we can beat best_k
        # If we pass best_k + 1, we update and continue
        check_limit = min(200, best_k + 1)
        k = check_window(p, p_sq, check_limit)
        
        if k > best_k:
            best_k = k
            best_p = p
            # If we found a longer run, we can try to extend it
            if best_k < 200:
                # Continue checking from best_k + 1
                extended = check_window(p, p_sq, 200)
                if extended > best_k:
                    best_k = extended
                    best_p = p
    
    return {'p': best_p, 'k': best_k}
