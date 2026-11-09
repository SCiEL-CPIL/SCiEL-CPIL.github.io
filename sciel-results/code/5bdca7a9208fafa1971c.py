"""Program for: Long runs after p^2 with no prime factor exceeding p (p <= 10^9).
Optimized search with fast p-smooth checking and broad prime coverage."""

import time
import math
import random

math_gcd = math.gcd

# ---------- Deterministic Miller-Rabin ----------
def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    # Bases sufficient for n < 2^64
    for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if a % n == 0:
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

# ---------- Pollard's rho ----------
def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        c = random.randrange(1, n)
        x = random.randrange(2, n)
        y = x
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math_gcd(abs(x - y), n)
        if d != n:
            return d

# ---------- Factor with early cap check ----------
def all_factors_leq(n, cap):
    """Return True if all prime factors of n are <= cap."""
    if n <= 1:
        return True
    
    # Trial divide by small primes up to min(cap, 100)
    # This handles most small factors quickly
    v = n
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
    for d in small_primes:
        if d > cap:
            break
        while v % d == 0:
            v //= d
            
    if v == 1:
        return True
        
    if v <= cap:
        # v is a single prime or 1; if v > 1 it's prime (since we divided by all primes up to 97)
        # But v could be composite with all factors > 97. Check primality.
        # If v is prime, it must be <= cap.
        # If v is composite, its factors are > 97. If v <= cap, then factors <= v <= cap.
        # So if v <= cap, it's always true? 
        # Wait. If v is composite and v <= cap, then all prime factors of v are <= v <= cap. 
        # So yes, if v <= cap, return True.
        return True

    # v > cap
    # If v is prime, then v > cap, so fail.
    if is_prime(v):
        return False
        
    # v is composite and v > cap.
    # We need to check if it has a factor > cap.
    # Factor v.
    # Since v > cap, and we removed small factors, v is likely a product of 2 primes > 97.
    # If v = a * b, and a <= cap, b <= cap, then v <= cap^2.
    # If v > cap^2, then at least one factor is > cap.
    if v > cap * cap:
        return False
        
    # v <= cap^2 and v > cap.
    # Try to find a factor.
    # If we find a factor d, check d and v//d.
    d = pollard_rho(v)
    if d == v:
        # Should not happen if composite, but if it does, treat as prime
        return is_prime(v) and v <= cap
    return all_factors_leq(d, cap) and all_factors_leq(v // d, cap)

# ---------- Sieve of primes ----------
def sieve_primes(limit):
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, limit + 1) if is_p[i]]

# ---------- Run-length evaluation ----------
def compute_k(p, max_k=200):
    """Compute the length of the run starting at p^2+1 where all prime factors <= p."""
    base = p * p
    k = 0
    for i in range(1, max_k + 1):
        if not all_factors_leq(base + i, p):
            break
        k += 1
    return k

# ---------- Main construct ----------
def construct():
    deadline = time.time() + 110  # 110s safety margin before 120s kill
    best_p, best_k = 2, 0
    
    # Generate primes up to a reasonable limit
    # 200,000 primes allows p^2 ~ 4*10^10, which is fast to factor
    primes_small = sieve_primes(200000)
    
    # Pass 1: all small primes
    for p in primes_small:
        if time.time() > deadline:
            break
        k = compute_k(p, 200)
        if k > best_k:
            best_k = k
            best_p = p
            
    # If we found a good k, we can stop. 
    # If not, we can try larger primes, but factorization gets slower.
    # Given the stagnation note, a structural change to a dense search of small primes is safer.
    
    return {'p': best_p, 'k': best_k}
