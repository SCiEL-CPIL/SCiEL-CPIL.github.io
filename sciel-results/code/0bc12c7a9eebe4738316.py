import math
import time
import random
import sys

# Safety margin: evaluator kills at 150s. We stop at 140s to be safe.
DEADLINE = time.time() + 140.0

def is_prime(n):
    """Deterministic Miller-Rabin for n < 3,474,749,660,383"""
    if n < 2:
        return False
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
        
    witnesses = [2, 3, 5, 7, 11, 13, 17]
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

def pollard_brent(n):
    """Pollard's Rho algorithm with Brent's cycle detection."""
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    
    while True:
        y, c, m = random.randint(1, n - 1), random.randint(1, n - 1), random.randint(1, n - 1)
        g, q, r, x, ys = 1, 1, 1, 0, y
        while g == 1:
            x = y
            for i in range(r):
                y = (y * y + c) % n
            k = 0
            while k < r and g == 1:
                ys = y
                for i in range(min(m, r - k)):
                    y = (y * y + c) % n
                    q = (q * abs(x - y)) % n
                g = math.gcd(q, n)
                k += m
            r += m
        if g == n:
            while True:
                ys = (ys * ys + c) % n
                g = math.gcd(abs(x - ys), n)
                if g != 1:
                    break
        if g != n:
            return g

def factorize(n, factors):
    """Recursively factorize n into a list of prime factors."""
    if n == 1:
        return
    if is_prime(n):
        factors.append(n)
        return
    
    d = pollard_brent(n)
    factorize(d, factors)
    factorize(n // d, factors)

def largest_prime_factor(n):
    """Return the largest prime factor of n."""
    if n == 1:
        return 1
    factors = []
    factorize(n, factors)
    return max(factors) if factors else 1

def construct():
    best_k = 0
    best_p = 2
    start_time = time.time()
    
    # We will search primes p in increasing order.
    # For each p, we check if p^2 + 1 ... p^2 + k are p-smooth.
    # Condition: No prime factor of (p^2 + i) is greater than p.
    # Equivalently: For all primes q > p, q does not divide p^2 + i for any i in [1, k].
    
    # Optimization Strategy:
    # Instead of factoring every number for every p (expensive), we can pre-compute
    # which primes q > p could potentially divide p^2 + i for small i.
    # However, p varies.
    
    # Alternative Strategy (Sieve-based rejection):
    # For a fixed p, if there exists a prime q > p such that q | (p^2 + i) for some i <= K,
    # then the run length k is at most i-1 (if no other q' > p divides earlier terms).
    # Actually, if q | (p^2 + i), then p^2 = -i mod q.
    # This means p is a square root of -i mod q.
    # This only has a solution if (-i/q) = 1.
    
    # Let's stick to the direct check but optimize the "failure" detection.
    # A number N is NOT p-smooth if it has a prime factor > p.
    # If N is prime and N > p, it fails.
    # If N is composite, we need to check its factors.
    
    # To speed up, we can first check if N has any small prime factor.
    # But the bottleneck is usually finding a LARGE prime factor or proving N is prime.
    
    # Let's try a different search order or heuristic.
    # Small p are easy to factor. Large p are hard.
    # But large p have larger "room" for smooth numbers?
    # Actually, the density of p-smooth numbers near p^2 is roughly related to the Dickman function.
    # For x = p^2, y = p. u = log(x)/log(y) = 2.
    # The probability that a number near p^2 is p-smooth is roughly rho(2) = 1 - log(2) ≈ 0.3.
    # So we expect runs of length ~ 1/0.3 = 3.
    # To get k=200, we need an incredibly lucky streak, which is unlikely for random p.
    # Wait, the problem says "Long runs ... are finite evidence".
    # It implies such p exist.
    
    # Let's just iterate p and check k.
    # To make it faster, we can skip p if p^2+1 is not p-smooth.
    # p^2+1 is not p-smooth if it has a prime factor > p.
    # p^2+1 is often prime or has large factors.
    
    # Let's optimize the inner loop.
    # We can pre-compute primes up to a certain limit for trial division?
    # No, p can be 10^9.
    
    # Let's use a sieve to find candidates p.
    # If p is such that p^2 + 1 is divisible by a prime q > p, it fails immediately.
    # How to find such q quickly?
    # We can't easily enumerate q.
    
    # Let's just run the factorization but with a time check.
    # If we spend too much time on one p, we might miss better ones.
    # But factorization of p^2+1 (size ~ 10^18) is fast with Pollard Rho (ms).
    # The bottleneck is if we have to factor many numbers for many p.
    
    # Let's try to find p where p^2+1 is "smooth".
    # We can search for p such that p^2+1 has small factors.
    
    p = 3
    # We will check p, p+2, p+4...
    # If p is not prime, skip.
    
    while time.time() < DEADLINE:
        if is_prime(p):
            k = 0
            # Check run length
            # We can break early if k reaches 200.
            while k < 200:
                val = p * p + k + 1
                # Check if val is p-smooth.
                # If val is prime and val > p, fail.
                # If val is composite, check largest factor.
                
                # Optimization:
                # If val <= p, it's smooth. (Unlikely for p^2+1)
                # If val is prime:
                if is_prime(val):
                    if val > p:
                        break
                    else:
                        k += 1
                        continue
                
                # If val is composite:
                # We need to check if it has a factor > p.
                # We can try to factor it.
                # But factorization is expensive.
                # Can we do a quick check?
                # If val has a factor d such that d > p, then val/d < val/p = p + (k+1)/p < p+1.
                # So if val has a factor d > p, the co-factor val/d is < p.
                # This means val has a small factor < p.
                # Wait. If d > p, then val/d < val/p.
                # Since val = p^2 + k + 1, val/p = p + (k+1)/p.
                # So val/d < p + 1.
                # So if we find a factor d of val such that p < d < val, then val is not p-smooth.
                # Conversely, if all factors d of val are <= p, then val is p-smooth.
                
                # So, to check if val is p-smooth:
                # 1. Find all prime factors of val.
                # 2. Check if max(factor) <= p.
                
                # Can we do this faster?
                # We can trial divide by primes up to p? No, p is large.
                # We can use Pollard Rho.
                
                lpf = largest_prime_factor(val)
                if lpf > p:
                    break
                k += 1
            
            if k > best_k:
                best_k = k
                best_p = p
                if best_k == 200:
                    break
        
        p += 2
        
    return {'p': best_p, 'k': best_k}

# Test
if __name__ == "__main__":
    result = construct()
    print(result)
