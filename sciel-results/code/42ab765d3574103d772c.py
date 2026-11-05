import math
import time
import random

def construct():
    deadline = time.time() + 115.0
    
    def sieve(limit):
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0:2] = b'\x00\x00'
        for i in range(2, int(limit**0.5) + 1):
            if is_p[i]:
                is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]
    
    # Sieve primes up to 100000 for trial division and prime generation
    SIEVE_LIMIT = 100000
    primes = sieve(SIEVE_LIMIT)
    small_primes = [p for p in primes if p <= 1000]
    
    def is_prime_mr(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        small_primes_mr = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
        for sp in small_primes_mr:
            if n == sp:
                return True
            if n % sp == 0:
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
    
    def is_p_smooth_fast(n, p, small_primes, all_primes):
        """Check if n is p-smooth using fast trial division by small primes first."""
        if n <= 1:
            return True
        # First, divide out small primes
        for q in small_primes:
            if q > p:
                break
            if q * q > n:
                break
            while n % q == 0:
                n //= q
        if n <= 1:
            return True
        if n <= p:
            return True
        # n > p and n is not 1. Check if n is prime (then it's a factor > p, fail)
        # or composite with all factors <= p
        # If n is prime and n > p, fail
        if is_prime_mr(n):
            return n <= p
        # n is composite. Need to check if all its prime factors are <= p.
        # Trial divide by remaining primes <= p
        for q in all_primes:
            if q > p:
                break
            if q * q > n:
                break
            if q <= 1000:  # already done
                continue
            while n % q == 0:
                n //= q
            if n <= 1:
                return True
        if n > 1:
            if n <= p:
                return True
            return is_prime_mr(n) and n <= p
        return True
    
    best_k = 0
    best_p = 2
    
    # Check p = 2
    k = 0
    for k_val in range(1, 201):
        v = 4 + k_val
        if is_p_smooth_fast(v, 2, small_primes, primes):
            k = k_val
        else:
            break
    if k > best_k:
        best_k = k
        best_p = 2
    
    # Check primes in increasing order, but with time budget
    for p in primes:
        if time.time() > deadline:
            break
        if p > 10**9:
            break
        if p < 2:
            continue
        
        k = 0
        p_sq = p * p
        for k_val in range(1, 201):
            if time.time() > deadline:
                break
            v = p_sq + k_val
            if is_p_smooth_fast(v, p, small_primes, primes):
                k = k_val
            else:
                break
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # Try some larger primes using Miller-Rabin
    random.seed(42)
    checked = set(primes)
    for _ in range(5000):
        if time.time() > deadline:
            break
        candidate = random.randint(100000, 1000000)
        if candidate in checked:
            continue
        checked.add(candidate)
        if not is_prime_mr(candidate):
            continue
        p = candidate
        k = 0
        p_sq = p * p
        for k_val in range(1, 201):
            if time.time() > deadline:
                break
            v = p_sq + k_val
            if is_p_smooth_fast(v, p, small_primes, primes):
                k = k_val
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p
    
    return {'p': best_p, 'k': best_k}
