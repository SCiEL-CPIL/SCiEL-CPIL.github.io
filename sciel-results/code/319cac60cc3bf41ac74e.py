import random
import math

# Miller-Rabin Primality Test
def is_prime(n):
    if n < 2:
        return False
    # Small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n % p == 0:
            return n == p
    
    # Write n-1 as d*2^r
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    
    # Witnesses for 32-bit and 64-bit integers
    # For n < 3,474,749,660,383, these bases are sufficient
    # Our n <= 10^12, so this is more than enough.
    witnesses = [2, 3, 5, 7, 11, 13]
    
    for a in witnesses:
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

# Pollard's Rho Algorithm
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

def factor(n, result=None):
    if result is None:
        result = []
    if n == 1:
        return result
    if is_prime(n):
        result.append(n)
        return result
    
    d = pollard_rho(n)
    factor(d, result)
    factor(n // d, result)
    return result

def phi(n):
    if n == 1:
        return 1
    factors = factor(n)
    # Get unique prime factors
    unique_primes = set(factors)
    result = n
    for p in unique_primes:
        result -= result // p
    return result

def construct():
    # Set seed for reproducibility in random choices if needed, 
    # but deterministic logic is preferred for "construct"
    # We will use a mix of deterministic candidates and random ones.
    
    # Pre-compute nothing, rely on fast factorization.
    
    # Try k from max down to 1
    for k in range(200, 0, -1):
        # Candidate starts
        # We want to avoid small numbers where collisions are frequent.
        # Large numbers are better.
        
        candidates = set()
        
        # Deterministic candidates near the upper bound
        max_start = 10**12 - k + 1
        candidates.add(max_start)
        candidates.add(max_start - 1000)
        candidates.add(max_start - 10000)
        
        # Mid-range candidates
        candidates.add(10**12 // 2)
        candidates.add(10**11)
        candidates.add(10**10)
        
        # Random candidates (fixed seed for stability across runs if desired, 
        # or just use system random. Since we want to maximize k, 
        # random sampling in high regions is good.)
        # We'll add a few random points in [10^10, 10^12]
        for _ in range(10):
            s = random.randint(10**10, max_start)
            candidates.add(s)
            
        # Filter valid candidates
        valid_candidates = [s for s in candidates if 1 <= s <= max_start]
        
        # Shuffle to try random ones first? No, deterministic ones near max are strong.
        # But maybe random ones hit a "good" spot faster.
        # Let's try them in a specific order: High deterministic, then random.
        
        for s in valid_candidates:
            # Check if phi(s...s+k-1) are distinct
            seen = set()
            valid = True
            for i in range(k):
                val = phi(s + i)
                if val in seen:
                    valid = False
                    break
                seen.add(val)
            
            if valid:
                return {'start': s, 'length': k}
        
        # If no candidate worked for this k, try a denser search?
        # For now, we just proceed to k-1.
        # If k=200 fails, k=199 is tried.
        
    # Fallback
    return {'start': 1, 'length': 1}
