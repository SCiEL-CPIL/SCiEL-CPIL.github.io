import time
import math
import random
import sys

# Increase recursion limit just in case, though we use iterative stack
sys.setrecursionlimit(10000)

# Deterministic Miller-Rabin for 64-bit integers
# Bases: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37
# Or the specific set for < 2^64: 2, 325, 9375, 28178, 450775, 9780504, 1795265022
_MR_BASES_64 = [2, 325, 9375, 28178, 450775, 9780504, 1795265022]

def is_prime(n):
    if n < 2:
        return False
    # Small primes check
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
        
    for a in _MR_BASES_64:
        if a % n == 0:
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

def pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randint(2, n - 1)
        c = random.randint(1, n - 1)
        m = 128
        g = r = q = 1
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

def is_p_smooth(n, p):
    """
    Check if all prime factors of n are <= p.
    Uses trial division for small primes, then Pollard-Brent.
    Returns False immediately if a prime factor > p is found.
    """
    if n <= 1:
        return True
    
    # Trial division by small primes up to 100
    # This is very fast and removes small factors
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for q in small_primes:
        if n % q == 0:
            while n % q == 0:
                n //= q
            if n == 1:
                return True
    
    if n == 1:
        return True
        
    # If n is prime, check if it's <= p
    if is_prime(n):
        return n <= p
        
    # Otherwise, n is composite.
    # We need to ensure all its factors are <= p.
    # Use a stack to factorize n.
    stack = [n]
    while stack:
        x = stack.pop()
        if x == 1:
            continue
        if is_prime(x):
            if x > p:
                return False
            continue
        
        # Find a factor d of x
        d = pollard_brent(x)
        if d == x:
            # Pollard failed, retry or handle error
            # For robustness, just retry pollard
            continue
            
        # Check if d is > p and prime?
        # Actually, we just push d and x//d to stack.
        # The is_prime check in the loop will catch if d > p.
        # But we can optimize: if d > p, we don't need to factor d fully if d is prime.
        # But we don't know if d is prime yet.
        # So just push.
        stack.append(d)
        stack.append(x // d)
        
    return True

def run_length(p, max_k=200):
    """
    Calculate the run length for prime p.
    """
    p_sq = p * p
    k = 0
    for i in range(1, max_k + 1):
        n = p_sq + i
        if not is_p_smooth(n, p):
            break
        k += 1
    return k

def sieve(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i*i, n + 1, i):
                is_prime[j] = False
    return [i for i, prime in enumerate(is_prime) if prime]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    best_k = 0
    best_p = 2
    
    # Phase 1: Small Primes (Dense Scan)
    # Primes up to 2,000,000
    # Sieve is fast
    limit1 = 2000000
    primes1 = sieve(limit1)
    
    for p in primes1:
        if time.time() > deadline - 60: # Leave time for other phases
            break
        k = run_length(p)
        if k > best_k:
            best_k = k
            best_p = p
            
    # Phase 2: Medium Primes (Sparse Scan)
    # Primes between 2,000,000 and 10,000,000
    # We can generate them or use a segmented sieve.
    # For simplicity, we can just iterate and check primality, but that's slow.
    # Better: Sieve up to 10,000,000 and take the tail.
    # Sieve(10^7) takes ~1s and 10MB. Acceptable.
    if time.time() < deadline - 40:
        limit2 = 10000000
        primes2 = sieve(limit2)
        # Only check primes > limit1
        for p in primes2:
            if time.time() > deadline - 40:
                break
            if p <= limit1:
                continue
            k = run_length(p)
            if k > best_k:
                best_k = k
                best_p = p
                
    # Phase 3: Large Primes (Random Sampling)
    # Sample primes in [10^7, 10^9]
    # We need to find primes.
    # Random candidate, check is_prime, then run_length.
    if time.time() < deadline - 10:
        random.seed(42) # Deterministic for reproducibility if needed, but random is fine
        # Try to find ~100 large primes
        attempts = 0
        while time.time() < deadline - 5 and attempts < 500:
            attempts += 1
            # Generate random odd number
            # Bias towards larger numbers? Uniform is fine.
            candidate = random.randint(10**7, 10**9)
            if candidate % 2 == 0:
                candidate += 1
            
            # Find next prime
            while not is_prime(candidate):
                candidate += 2
                if candidate > 10**9:
                    break
            
            if candidate > 10**9:
                continue
                
            # Optional: Skip if we've likely checked it (not strictly necessary)
            k = run_length(candidate)
            if k > best_k:
                best_k = k
                best_p = candidate
                
    # Fallback
    if best_k < 1:
        # Find a valid p, k=1
        # p=7, p^2+1=50=2*5*5. All factors <= 7.
        best_p = 7
        best_k = 1

    return {'p': best_p, 'k': best_k}

# To ensure the module is loaded correctly if imported
if __name__ == "__main__":
    # For local testing
    res = construct()
    print(res)
