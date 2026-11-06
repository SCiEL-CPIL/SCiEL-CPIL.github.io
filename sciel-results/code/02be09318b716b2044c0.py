import time
import random
from math import gcd, isqrt
from functools import lru_cache

def is_prime(n):
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # Deterministic witnesses for 64-bit integers
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
            d = gcd(abs(x - y), n)
        if d != n:
            return d

def factor(n, max_factor=None):
    """Factor n, optionally stopping early if a factor > max_factor is found."""
    if n == 1:
        return []
    factors = []
    # Small primes trial division
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        while n % p == 0:
            factors.append(p)
            n //= p
            if max_factor is not None and p > max_factor:
                return factors
    if n == 1:
        return factors
    if is_prime(n):
        if max_factor is not None and n > max_factor:
            factors.append(n)
            return factors
        factors.append(n)
        return factors
    
    # Pollard Rho for remaining composite
    stack = [n]
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            if max_factor is not None and m > max_factor:
                factors.append(m)
                return factors
            factors.append(m)
        else:
            d = pollard_rho(m)
            if d == m:
                # Shouldn't happen, but handle it
                if max_factor is not None and m > max_factor:
                    factors.append(m)
                    return factors
                factors.append(m)
            else:
                stack.append(d)
                stack.append(m // d)
    return factors

def sieve(n):
    is_prime_arr = [True] * (n + 1)
    is_prime_arr[0] = is_prime_arr[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime_arr[i]:
            for j in range(i*i, n + 1, i):
                is_prime_arr[j] = False
    return [i for i, prime in enumerate(is_prime_arr) if prime]

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s margin
    
    # Sieve primes up to 10^7 for candidate p and trial division
    limit = 10**7
    primes = sieve(limit)
    
    # Precompute small primes up to 10^6 for smoothness checking via interval sieve
    small_primes_limit = 10**6
    small_primes = sieve(small_primes_limit)
    
    best_k = 0
    best_p = 2
    
    # We iterate through primes p
    # If p is small, p^2 is small.
    # If p is large, p^2 is large.
    
    for idx, p in enumerate(primes):
        if time.time() > deadline:
            break
        
        # Optional: Skip very small p if we already have a good k
        # For now, just check all.
        
        p2 = p * p
        K = 200
        
        # Batched verification
        # We want to find the max k such that p^2+1 ... p^2+k are p-smooth.
        
        # Initialize array of values and validity
        vals = [p2 + i for i in range(1, K + 1)]
        valid = [True] * K
        
        # 1. Trial division by small primes q <= p
        # If q <= p, we remove factors of q from vals.
        # If q > p, we mark vals divisible by q as invalid.
        
        # We only need to check q up to sqrt(max_val) ~ p.
        # But we only have small_primes up to 10^6.
        # If p > 10^6, we can't remove all factors q <= p using small_primes.
        
        # Strategy:
        # Divide out all small primes q <= min(p, 10^6).
        # For q > p (from our small_primes list), mark invalid.
        # Remaining numbers may still have factors > p.
        
        max_q_to_check = min(p, small_primes_limit)
        
        for q in small_primes:
            if q > max_q_to_check:
                # If q > p, then any val divisible by q has a factor > p -> Invalid
                if q > p:
                    for i in range(K):
                        if valid[i] and vals[i] % q == 0:
                            valid[i] = False
                else:
                    # This case shouldn't happen if loop condition is q <= max_q_to_check
                    pass
                continue
            
            # q <= p and q <= small_primes_limit
            # Remove factors of q from valid vals
            for i in range(K):
                if valid[i] and vals[i] % q == 0:
                    while vals[i] % q == 0:
                        vals[i] //= q
            
        # Now check the remaining vals
        # For each valid val, we need to ensure it has no prime factor > p.
        # Current val is the cofactor after removing primes <= min(p, 10^6).
        
        k = 0
        for i in range(K):
            if not valid[i]:
                break
            
            n = vals[i]
            if n == 1:
                k += 1
                continue
                
            # Check if n is prime
            if is_prime(n):
                if n <= p:
                    k += 1
                else:
                    break
            else:
                # n is composite
                # Factor it and check max factor
                facs = factor(n, max_factor=p)
                if not facs:
                    # Should not happen for n > 1
                    k += 1
                else:
                    if max(facs) > p:
                        break
                    else:
                        k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
            
    return {'p': best_p, 'k': best_k}
