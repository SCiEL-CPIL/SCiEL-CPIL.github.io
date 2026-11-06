import math
import time
import random
import sys

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 145  # 5s margin from 150s limit
    
    # Deterministic Miller-Rabin bases for n < 2^64
    # For n < 3,474,749,660,383, bases 2,3,5,7,11,13,17 are enough.
    # For n < 341,550,071,728,321, bases 2,3,5,7,11,13,17,19,23 are enough.
    # We are dealing with n up to ~ 10^18 (p^2 + 200 for p=10^9).
    # Actually p <= 10^9, so p^2 <= 10^18.
    # Standard set for < 2^64: [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    
    def is_prime(n):
        if n < 2:
            return False
        # Small primes check
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n == p:
                return True
            if n % p == 0:
                return False
        
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
        
        # Bounded iterations to prevent infinite loops on difficult composites
        # Use random restarts
        while True:
            x = random.randint(2, n - 1)
            y = x
            c = random.randint(1, n - 1)
            d = 1
            # Limit iterations per attempt to avoid hanging on specific hard cases
            for _ in range(10000):
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
                if d != 1:
                    break
            if d != n:
                return d
            # If d == n, retry with different x, c

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
            if x == 1:
                continue
            if is_prime(x):
                factors.append(x)
            else:
                d = pollard_rho(x)
                stack.append(d)
                stack.append(x // d)
        return factors

    def check_run(p, max_k=200):
        """
        Checks p^2 + 1 ... p^2 + max_k.
        Returns the length of the run where all prime factors are <= p.
        """
        k = 0
        p2 = p * p
        for offset in range(1, max_k + 1):
            n = p2 + offset
            # Optimization: If n is prime and n > p, it fails immediately.
            # is_prime is much faster than full factorization.
            if is_prime(n):
                if n > p:
                    break
                else:
                    k += 1
                    continue
            
            # If composite, we need to ensure no factor > p.
            # We can factorize and check max.
            # Alternatively, we can try to find a factor > p?
            # Factorization is safer.
            factors = factorize(n)
            if max(factors) > p:
                break
            else:
                k += 1
        return k

    best_p = 7
    best_k = 1 # Fallback known valid
    
    # 1. Check a few small known good candidates quickly?
    # The current best score is 2.333, which implies k=7 for some p.
    # Let's just rely on the random search to find better.
    
    # Strategy: Random sampling of primes in [10, 10^9]
    # We will sample candidates. For each prime candidate, we check the run.
    # To save time, we prioritize larger p? Or uniform?
    # Larger p generally allows for longer runs because the "gap" to the next prime factor is relatively smaller?
    # Actually, the density of p-smooth numbers decreases as p grows?
    # No, for a fixed n, the probability that all factors are <= p increases as p increases.
    # But here n ~ p^2.
    # Let's just do uniform random sampling in log space or linear space.
    # Linear space up to 10^9.
    
    # Number of attempts
    # Each check of a prime p involves checking up to 200 numbers.
    # Most will fail at offset 1 or 2.
    # is_prime is ~ microsecond. factorize is ~ millisecond.
    # We can probably do 1000-5000 checks in 145s.
    
    attempts = 0
    while time.time() - start_time < DEADLINE:
        attempts += 1
        
        # Generate random candidate
        # Bias towards larger p? 
        # Let's try uniform in [100, 10^9]
        # Using random.randint
        candidate = random.randint(100, 10**9)
        
        # Ensure it's odd
        if candidate % 2 == 0:
            candidate += 1
            
        # Check if prime
        if not is_prime(candidate):
            continue
            
        p = candidate
        
        # Check run
        # If p is very large, factorization of p^2+1 might be harder?
        # p^2+1 is ~ 10^18. Pollard Rho handles this well.
        
        k = check_run(p, 200)
        
        if k > best_k:
            best_k = k
            best_p = p
            # If we find a very good k, we might want to search around it?
            # Or just continue random.
            # Let's print to stderr for debugging if needed, but keep clean.
            
    # Final sanity check: ensure best_k >= 1
    if best_k < 1:
        # This should not happen as p=7, k=1 is always valid
        # p=7: 49+1=50=2*5^2 (max 5 <= 7). 49+2=51=3*17 (17>7). So k=1.
        best_p = 7
        best_k = 1

    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
