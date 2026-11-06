import time
import random
from math import gcd
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
                g = gcd(q, n)
                k += m
            r *= 2
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = gcd(abs(x - ys), n)
                if g > 1:
                    break
        if g != n:
            return g

def factor(n, max_factor=None):
    """Factor n, optionally stopping early if a factor > max_factor is found."""
    if n == 1:
        return []
    factors = []
    # Small primes trial division
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for p in small_primes:
        if p > max_factor:
            break
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
    
    # Pollard Brent for remaining composite
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
            d = pollard_brent(m)
            if d == m:
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
    deadline = start_time + 115  # 5s margin for 120s limit
    
    # Sieve primes up to 10^7 for initial scan
    limit = 10**7
    primes = sieve(limit)
    
    best_k = 0
    best_p = 2
    
    # Helper to check if n has any prime factor > p
    def has_large_factor(n, p):
        if n <= 1:
            return False
        if is_prime(n):
            return n > p
        # Trial divide by small primes
        small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
        for sp in small_primes:
            if sp > p:
                break
            if n % sp == 0:
                while n % sp == 0:
                    n //= sp
                if n == 1:
                    return False
        if n == 1:
            return False
        if is_prime(n):
            return n > p
        # Use Pollard-Brent to find a factor
        d = pollard_brent(n)
        if d == n:
            return n > p
        return has_large_factor(d, p) or has_large_factor(n // d, p)
    
    # First pass: linear scan of small primes
    for p in primes:
        if time.time() > deadline:
            break
        
        p2 = p * p
        
        # Quick check: if p^2 + 1 is prime, skip
        n0 = p2 + 1
        if is_prime(n0):
            if n0 > p:
                continue
        
        # Factor p^2 + 1 with early termination
        facs = factor(n0, max_factor=p)
        if not facs or max(facs) > p:
            continue
        
        k = 1
        # Check further j
        while k < 200:
            if time.time() > deadline:
                break
            
            n = p2 + k + 1
            
            # Quick primality check
            if is_prime(n):
                if n > p:
                    break
                else:
                    k += 1
                    continue
            
            # Factor with early termination
            facs = factor(n, max_factor=p)
            if not facs or max(facs) > p:
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # Second pass: random search for larger primes
    # Focus on primes in range [10^7, 10^9]
    random.seed(42)
    checked = set(primes)
    
    while time.time() < deadline - 5:
        # Generate random odd number in range [10^7, 10^9]
        if random.random() < 0.5:
            # Try near 10^7
            p_candidate = random.randint(10**7, 2 * 10**7)
        else:
            # Try in [10^7, 10^9]
            p_candidate = random.randint(10**7, 10**9)
        
        # Make it odd
        if p_candidate % 2 == 0:
            p_candidate += 1
        
        if p_candidate in checked:
            continue
        checked.add(p_candidate)
        
        # Check if prime
        if not is_prime(p_candidate):
            # Find next prime
            p_candidate += 2
            while not is_prime(p_candidate):
                p_candidate += 2
                if p_candidate > 10**9:
                    break
            if p_candidate > 10**9 or p_candidate in checked:
                continue
            checked.add(p_candidate)
        
        p = p_candidate
        p2 = p * p
        
        # Quick check: if p^2 + 1 is prime, skip
        n0 = p2 + 1
        if is_prime(n0):
            if n0 > p:
                continue
        
        # Factor p^2 + 1 with early termination
        facs = factor(n0, max_factor=p)
        if not facs or max(facs) > p:
            continue
        
        k = 1
        # Check further j
        while k < 200:
            if time.time() > deadline - 2:
                break
            
            n = p2 + k + 1
            
            # Quick primality check
            if is_prime(n):
                if n > p:
                    break
                else:
                    k += 1
                    continue
            
            # Factor with early termination
            facs = factor(n, max_factor=p)
            if not facs or max(facs) > p:
                break
            k += 1
        
        if k > best_k:
            best_k = k
            best_p = p
    
    return {'p': best_p, 'k': best_k}
