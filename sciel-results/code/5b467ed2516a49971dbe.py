import math
import time
import random
import sys

# Deterministic Miller-Rabin bases for n < 2^64
MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def is_prime(n):
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in MR_BASES:
        if a >= n:
            continue
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(s - 1):
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
        y = random.randrange(2, n)
        c = random.randrange(1, n)
        m = random.randrange(1, n)
        g = 1
        r = 1
        q = 1
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

def factorize(n, limit=None):
    """
    Returns list of prime factors.
    If limit is provided, we only care if all factors are <= limit.
    Returns None if a factor > limit is found.
    """
    if n == 1:
        return []
    
    # Quick small prime trial division
    # We can hardcode a few small primes for speed
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    
    def _factorize_rec(m, limit):
        if m == 1:
            return []
        if limit is not None and m > limit:
            # Check if m is prime
            if is_prime(m):
                return [m]
            # If m is composite and > limit, it might have a factor > limit
            # But wait, if m > limit and we need ALL factors <= limit, 
            # then if m is prime, it fails. If m is composite, it could still fail if it has a prime factor > limit.
            # Actually, if m > limit, and we want all factors <= limit, 
            # we must fully factor m and check each.
            # However, if m is prime, it is > limit, so return [m] which will be caught by the check.
            pass
        
        # Trial division by small primes
        for p in small_primes:
            if p * p > m:
                break
            if m % p == 0:
                if limit is not None and p > limit:
                    return [p] # Failed
                factors = [p]
                rest = m // p
                while rest % p == 0:
                    factors.append(p)
                    rest //= p
                rest_factors = _factorize_rec(rest, limit)
                if rest_factors is None:
                    return None
                return factors + rest_factors
        
        if m == 1:
            return []
        
        # Check if m is prime
        if is_prime(m):
            if limit is not None and m > limit:
                return [m] # Failed
            return [m]
        
        # Pollard Brent
        d = pollard_brent(m)
        if d == m:
            # Retry or fallback
            return _factorize_rec(m, limit)
        factors1 = _factorize_rec(d, limit)
        if factors1 is None:
            return None
        factors2 = _factorize_rec(m // d, limit)
        if factors2 is None:
            return None
        return factors1 + factors2

    return _factorize_rec(n, limit)

def check_run(p, max_k=200, deadline=None):
    """
    Check how long the run is for prime p.
    Returns k such that p^2+1 ... p^2+k are all p-smooth.
    """
    k = 0
    start_val = p * p + 1
    for i in range(1, max_k + 1):
        if deadline is not None and time.time() > deadline:
            break
        val = start_val + i - 1
        # Check if val is p-smooth
        # We can do a quick check: if val > p and is_prime(val) and val > p: return k
        # But val might be composite.
        
        # Optimization: if val has a prime factor > p, stop.
        # We can try to find any factor > p.
        # If val is prime and > p, stop.
        if is_prime(val):
            if val > p:
                return k
            else:
                k += 1
                continue
        
        # If val is composite, we need to check if it has a factor > p.
        # Let's factorize it with a limit.
        # If factorize returns a factor > p, stop.
        # To speed up, we can check small factors first.
        # If val has a factor f <= p, we divide it out.
        # If the remaining part is > 1 and has a factor > p, stop.
        
        # Actually, the previous _factorize_rec is a bit slow if it doesn't short-circuit well.
        # Let's write a specific "is_p_smooth" function.
        
        if not is_p_smooth(val, p):
            return k
        k += 1
    return k

def is_p_smooth(n, p):
    """
    Returns True if all prime factors of n are <= p.
    """
    if n == 1:
        return True
    
    # Small prime trial division
    # We can use a precomputed list of primes up to some limit, or just the small ones.
    # For p up to 10^7, n is around 10^14.
    # Trial division up to sqrt(n) is too slow.
    
    # Strategy:
    # 1. Check if n is prime. If so, return n <= p.
    if is_prime(n):
        return n <= p
    
    # 2. Find a factor d.
    d = pollard_brent(n)
    if d == n:
        # Should not happen for composite, but just in case
        return is_p_smooth(n, p)
    
    return is_p_smooth(d, p) and is_p_smooth(n // d, p)

def sieve(n):
    """
    Sieve of Eratosthenes.
    Returns list of primes up to n.
    """
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10 second safety margin
    
    # 1. Generate primes up to a limit
    # Let's try 10^6 first, then expand if time permits.
    # Actually, the inspirations mention 10^7.
    # Sieve of 10^7 is fast in Python? ~0.1-0.5s.
    # Let's do 2 * 10^6 to be safe and fast, then maybe 10^7.
    
    # Let's start with a smaller set to ensure we get a result, then expand.
    # The seed program only checked up to 2000.
    # Let's check up to 10^5 first.
    
    primes_limit = 100000
    primes = sieve(primes_limit)
    
    best_k = 0
    best_p = 2
    
    # We can process primes in random order or descending?
    # Descending might find large p which might have long runs?
    # Or random to avoid bias.
    # Let's try random sampling from the sieve.
    
    # Actually, let's just iterate.
    # But we need to be careful with time.
    
    # Let's use a random subset to explore the space more broadly.
    random.seed(42)
    
    # We will test a sample of primes.
    # If we have time, we can test more.
    
    # Let's test all primes in the first batch, then if time remains, test more.
    
    # To maximize k, we should probably test larger primes?
    # Or does it not matter?
    # Let's just test a good number of primes.
    
    # Sort primes by some heuristic? 
    # Maybe primes that are close to 10^5?
    
    # Let's just iterate through the primes list.
    # If we run out of time, we return the best found.
    
    for p in primes:
        if time.time() > deadline:
            break
        
        # Check run length for p
        k = check_run(p, 200, deadline)
        
        if k > best_k:
            best_k = k
            best_p = p
            
            # If we found a very good one, we can stop?
            # Max k is 200. If we hit 200, we can stop.
            if best_k == 200:
                break
    
    # If we have time, we can try a larger sieve or random larger primes.
    # But generating a sieve up to 10^7 might take too long if we are already close to deadline.
    # Let's check time.
    
    if time.time() < deadline - 5:
        # Try a larger range
        # Let's try primes between 10^5 and 10^6
        # We can generate primes in this range using segmented sieve or just extend sieve.
        # Extending sieve to 10^6 is fast.
        
        primes_limit_2 = 1000000
        if primes_limit_2 > primes_limit:
            # We can just sieve again or extend.
            # Sieve again is simpler.
            primes_2 = sieve(primes_limit_2)
            # Only test primes > primes_limit
            for p in primes_2:
                if p <= primes_limit:
                    continue
                if time.time() > deadline:
                    break
                
                k = check_run(p, 200, deadline)
                if k > best_k:
                    best_k = k
                    best_p = p
                    if best_k == 200:
                        break
    
    return {'p': best_p, 'k': best_k}

# To ensure the function is available as per contract
# The seed program had construct() defined.

if __name__ == "__main__":
    res = construct()
    print(res)
