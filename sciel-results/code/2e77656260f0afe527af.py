"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random

# EVOLVE-BLOCK-START

# --- Prime utilities ---

def _is_prime(n):
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
    # Deterministic Miller-Rabin for n < 3,317,044,064,679,887,385,961,981
    for a in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
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


def _pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randrange(2, n - 1)
        c = random.randrange(1, n - 1)
        m = random.randrange(1, 100)
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


def _factorize(n):
    """Return list of prime factors of n (with multiplicity)."""
    if n <= 1:
        return []
    factors = []
    # Trial division by small primes
    small_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    for p in small_primes:
        while n % p == 0:
            factors.append(p)
            n //= p
    if n == 1:
        return factors
    # Recursive factorization
    stack = [n]
    while stack:
        x = stack.pop()
        if x == 1:
            continue
        if _is_prime(x):
            factors.append(x)
        else:
            d = _pollard_brent(x)
            stack.append(d)
            stack.append(x // d)
    return factors


def _largest_prime_factor(n):
    if n <= 1:
        return 1
    factors = _factorize(n)
    return max(factors) if factors else 1


def _is_p_smooth(n, p):
    """Check if all prime factors of n are <= p."""
    if n <= 1:
        return True
    # Quick check: if n itself is prime and n > p, return False
    if _is_prime(n) and n > p:
        return False
    factors = _factorize(n)
    return all(f <= p for f in factors)


# --- Search ---

def _generate_primes_up_to(limit):
    """Sieve of Eratosthenes."""
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]


def construct():
    start_time = time.time()
    deadline = 140.0  # 10s safety margin before 150s kill

    best_k = 0
    best_p = 2

    # Precompute primes up to a reasonable limit for initial search
    # We'll search primes in increasing order, but use a smarter approach:
    # For a given p, we need p^2+1, ..., p^2+k to all be p-smooth.
    # Key insight: p^2 + j for small j is ~ p^2. Its prime factors are typically
    # either small or large. For it to be p-smooth, all factors must be <= p.
    #
    # Strategy: Try many primes efficiently. For each prime p, check consecutive
    # values p^2+1, p^2+2, ... until we find one that is NOT p-smooth.
    #
    # To speed up: precompute primes up to some limit, then for larger p,
    # we can use the fact that p^2 + j has at most ~log(p^2) prime factors.

    # Phase 1: Try small primes with direct factorization
    # Phase 2: Try larger primes with optimized checks

    # First, let's try a range of primes. We'll iterate through primes and
    # for each, check the run length.
    
    # Generate primes up to 10^6 for efficient iteration
    primes_small = _generate_primes_up_to(1000000)
    
    for p in primes_small:
        if time.time() > deadline:
            break
        if p > 10**9:
            break
        # Check run length for this p
        k = 0
        for j in range(1, 201):
            if time.time() > deadline:
                break
            n = p * p + j
            if not _is_p_smooth(n, p):
                break
            k = j
        if k > best_k:
            best_k = k
            best_p = p
            # If we found a good run, we can potentially extend search
            # but for now just record it

    # Phase 2: Try larger primes (10^6 to 10^9) more sparsely
    # For large p, p^2 is huge (~10^18), so factorization is expensive.
    # But the probability that p^2+j is p-smooth for j up to k is low.
    # We'll sample primes in this range.
    
    # Actually, let's reconsider. The seed program only checked primes up to 2000
    # and found k=2. Let's see if we can do better with a smarter approach.
    
    # For the run to be long, we need p^2+1, p^2+2, ..., p^2+k to all be p-smooth.
    # Note that gcd(p^2+j, p) = gcd(j, p) = 1 (since p is prime and j < p for small j).
    # So p^2+j is coprime to p.
    
    # Let's try a few specific promising primes. Small primes like 2, 3, 5, 7, 11, 13, ...
    # might have longer runs because p-smooth means all factors <= p, and for small p,
    # p^2+j is close to p^2, so its factors tend to be around p.
    
    # Let's explicitly check small primes with a more thorough approach
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if time.time() > deadline:
            break
        k = 0
        for j in range(1, 201):
            if time.time() > deadline:
                break
            n = p * p + j
            if not _is_p_smooth(n, p):
                break
            k = j
        if k > best_k:
            best_k = k
            best_p = p

    # Try some medium primes
    medium_primes = [101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
                      211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317,
                      331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443,
                      449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577,
                      587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701]
    
    for p in medium_primes:
        if time.time() > deadline:
            break
        k = 0
        for j in range(1, 201):
            if time.time() > deadline:
                break
            n = p * p + j
            if not _is_p_smooth(n, p):
                break
            k = j
        if k > best_k:
            best_k = k
            best_p = p

    # If we still have time, try a random sample of larger primes
    if time.time() < deadline - 10:
        random.seed(42)
        # Sample primes in [10^6, 10^9]
        for _ in range(100):
            if time.time() > deadline:
                break
            p = random.randint(10**6, 10**9)
            # Find next prime
            while not _is_prime(p):
                p += 1
            if p > 10**9:
                continue
            k = 0
            for j in range(1, 50):  # Only check up to 50 for large primes (expensive)
                if time.time() > deadline:
                    break
                n = p * p + j
                if not _is_p_smooth(n, p):
                    break
                k = j
            if k > best_k:
                best_k = k
                best_p = p

    # Ensure at least k=1 is possible (p^2+1 must be p-smooth)
    # For p=2: 4+1=5, 5 is prime > 2, so not 2-smooth. 
    # For p=3: 9+1=10=2*5, 5>3, not 3-smooth.
    # For p=5: 25+1=26=2*13, 13>5, not 5-smooth.
    # For p=7: 49+1=50=2^2*5^2, all <=7. Good! k>=1.
    # So p=7, k>=1 is valid.
    
    if best_k < 1:
        # Fallback: find any valid p with k>=1
        for p in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
            if _is_p_smooth(p * p + 1, p):
                best_k = 1
                best_p = p
                break

    return {'p': best_p, 'k': best_k}

# EVOLVE-BLOCK-END
