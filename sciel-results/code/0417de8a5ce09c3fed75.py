import time
import random
import math

def is_prime(n):
    if n < 2:
        return False
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
    for p in small_primes:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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
    y, c, m = random.randrange(1, n), random.randrange(1, n), 128
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
    return g

def has_large_factor(n, limit):
    """Check if n has any prime factor > limit. Returns True if so."""
    if n <= 1:
        return False
    if n <= limit:
        return False
    
    # Quick check: if n is prime and n > limit, return True
    if is_prime(n):
        return n > limit
    
    # Use Pollard-Brent to find a factor
    d = pollard_brent(n)
    if d == n:
        # Failed to factor, treat as prime
        return n > limit
    
    # Check both factors
    if d > limit:
        return True
    if (n // d) > limit:
        # Need to check if n//d has a factor > limit
        return has_large_factor(n // d, limit)
    return False

def largest_prime_factor(n, limit):
    """Returns the largest prime factor of n, or limit+1 if there's a factor > limit."""
    if n <= 1:
        return 1
    if n <= limit:
        return n
    
    if is_prime(n):
        return n if n <= limit else limit + 1
    
    d = pollard_brent(n)
    if d == n:
        return n if n <= limit else limit + 1
    
    f1 = largest_prime_factor(d, limit)
    f2 = largest_prime_factor(n // d, limit)
    return max(f1, f2)

def segmented_sieve_window(p, k_max=200):
    """
    For a prime p, compute for each i in 1..k_max whether p^2+i has all prime factors <= p.
    Uses segmented sieve approach: divide out all small primes <= p from each number in the window.
    If any residual > 1 remains, that number has a factor > p.
    
    Returns the maximum k such that p^2+1, ..., p^2+k all have all prime factors <= p.
    """
    # We need to check numbers p^2+1, p^2+2, ..., p^2+k_max
    # For each, we want to know if it has any prime factor > p.
    
    # Strategy: For each prime q <= p, divide out q from all numbers in the window that are divisible by q.
    # After processing all primes q <= p, if any number still has residual > 1, it has a factor > p.
    
    # But p can be up to 10^9, so we can't iterate over all primes up to p.
    # Instead, we use a different approach:
    # For each number n = p^2 + i (1 <= i <= k_max), we check if it has a factor > p.
    # Since n < p^2 + k_max and p is large, n is much smaller than p^2 for large p.
    # Actually n ~ p^2, so n has factors that could be up to ~p.
    
    # Better approach: For each i, compute n = p^2 + i.
    # If n is prime and n > p, then it has a factor > p.
    # If n is composite, we need to check its factors.
    
    # The key insight: if n = p^2 + i and n has all factors <= p, then n is p-smooth.
    # We can check this by trial division by primes up to min(p, sqrt(n)) = p (since sqrt(p^2+i) < p+1 for small i).
    # But that's expensive.
    
    # Alternative: Use the fact that for n = p^2 + i, any prime factor q of n satisfies q <= p (if all factors <= p).
    # We can do trial division by primes up to some bound, and if a residual remains, use Pollard-Brent.
    
    # Let's use a hybrid: trial divide by small primes (up to 1000), then check if residual is 1 or has large factor.
    
    k = 0
    for i in range(1, k_max + 1):
        n = p * p + i
        # Check if n has all factors <= p
        if not _all_factors_leq(n, p):
            break
        k = i
    
    return k

def _all_factors_leq(n, limit):
    """Check if all prime factors of n are <= limit."""
    if n <= 1:
        return True
    if n <= limit:
        return True
    
    # Trial divide by small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for q in small_primes:
        if q > limit:
            break
        while n % q == 0:
            n //= q
    
    if n == 1:
        return True
    if n <= limit:
        return True
    
    # Check if n is prime
    if is_prime(n):
        return n <= limit
    
    # n is composite and > limit. Use Pollard-Brent to factor.
    def factorize(m, lim):
        if m == 1:
            return True
        if m <= lim:
            return True
        if is_prime(m):
            return m <= lim
        
        d = pollard_brent(m)
        if d == m:
            return m <= lim
        
        return factorize(d, lim) and factorize(m // d, lim)
    
    return factorize(n, limit)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin for 150s limit
    
    best_k = 0
    best_p = 2
    
    random.seed(42)
    
    # First, do a quick scan of small primes to establish a baseline
    for p in range(3, 5000, 2):
        if time.time() > deadline:
            break
        if not is_prime(p):
            continue
        
        k = 0
        for i in range(1, 201):
            n = p * p + i
            if not _all_factors_leq(n, p):
                break
            k = i
        
        if k > best_k:
            best_k = k
            best_p = p
    
    # Now do random search over larger primes
    # Key optimization: use segmented sieve approach to quickly reject candidates
    # where p^2+1 already has a large factor.
    
    while time.time() < deadline:
        # Pick a random candidate in a wide range
        # For larger p, the window p^2+1..p^2+k is more likely to have smooth numbers
        # because the numbers are larger and have more potential small factors.
        
        # Try different ranges
        if random.random() < 0.3:
            p = random.randint(10**4, 10**6)
        elif random.random() < 0.5:
            p = random.randint(10**6, 10**8)
        else:
            p = random.randint(10**8, 10**9)
        
        if p % 2 == 0:
            p += 1
        
        # Quick primality check
        if not is_prime(p):
            continue
        
        # Quick rejection: check if p^2+1 has all factors <= p
        # If not, skip this p entirely (k=0)
        n1 = p * p + 1
        if not _all_factors_leq(n1, p):
            continue
        
        # Now check the full window
        k = 1
        for i in range(2, 201):
            n = p * p + i
            if not _all_factors_leq(n, p):
                break
            k = i
        
        if k > best_k:
            best_k = k
            best_p = p
    
    return {'p': best_p, 'k': best_k}
