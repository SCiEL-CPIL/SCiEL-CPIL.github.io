import math
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 110  # 10s margin
    
    # Deterministic Miller-Rabin bases for n < 2^64
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0:
            return False
        if n % 3 == 0:
            return n == 3
        
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
            
        for a in MR_BASES:
            if a % n == 0:
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

    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        
        while True:
            x = random.randint(2, n - 1)
            y = x
            c = random.randint(1, n - 1)
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d
            # Retry if d == n

    def factorize(n):
        factors = []
        # Trial division by small primes
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                while n % p == 0:
                    factors.append(p)
                    n //= p
        if n == 1:
            return factors
        if is_prime(n):
            factors.append(n)
            return factors
        
        # Pollard Rho for remaining
        stack = [n]
        while stack:
            x = stack.pop()
            if is_prime(x):
                factors.append(x)
            else:
                d = pollard_rho(x)
                stack.append(d)
                stack.append(x // d)
        return factors

    def largest_factor_is_smaller_or_equal(n, p):
        if n == 1:
            return True
        if is_prime(n):
            return n <= p
        
        # Try to find a factor > p quickly?
        # Or just factorize and check max.
        # For n ~ p^2, factorization is fast.
        factors = factorize(n)
        return max(factors) <= p

    # Generate primes up to a certain limit
    # We need to iterate over candidate primes p.
    # Let's search p up to 10^7. 
    # 10^7 is ~ 664k primes. 
    # If each check takes 1ms, that's 664s. Too slow.
    # We need to be selective or faster.
    # 
    # Optimization: 
    # The "bad" number is likely prime. 
    # is_prime is very fast.
    # If n is composite, we factorize.
    # Most n will be prime or have small factors.
    
    # Let's limit search to p < 10^5 first to ensure we get a result, then expand if time permits.
    # Actually, let's just run a sieve up to 10^6 and iterate.
    
    limit = 1000000
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    
    primes = [i for i, is_p in enumerate(sieve) if is_p]
    
    best_k = 0
    best_p = 2 # Fallback
    
    for p in primes:
        if time.time() - start_time > DEADLINE:
            break
            
        if p > 10**9:
            break
            
        k = 0
        # Check p^2 + 1 to p^2 + K
        # We can cap K at 200 per task, but likely it's small.
        # To save time, we can break early if we find a bad one.
        for offset in range(1, 201):
            if time.time() - start_time > DEADLINE:
                break
            n = p * p + offset
            if largest_factor_is_smaller_or_equal(n, p):
                k += 1
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
            # If we find a large k, we might want to stop or continue?
            # Continue to see if we can find larger.
        
        # If k is 0, we still have a valid candidate? 
        # Task: 1 <= k. 
        # If we never find k>=1, we must return a valid one.
        # But we know p=7 gives k=1.
        # The loop will find it.

    # Ensure we return a valid result
    if best_k < 1:
        # Fallback to a known good prime if search failed (shouldn't happen)
        # p=7, k=1
        best_p = 7
        best_k = 1
        
    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
