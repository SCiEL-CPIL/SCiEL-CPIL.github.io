import time
import math
import random
import numpy as np

# ---------- Deterministic Miller-Rabin for 64-bit ----------
_MILLER_RABIN_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _is_prime_mr(n):
    if n < 2:
        return False
    for p in _MILLER_RABIN_BASES:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MILLER_RABIN_BASES:
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

# ---------- Pollard-Brent ----------
def _pollard_brent(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        y = random.randrange(1, n)
        c = random.randrange(1, n)
        m = random.randrange(1, n)
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
                if g != 1:
                    break
        if g != n:
            return g

def _factor(n, result):
    if n <= 1:
        return
    if _is_prime_mr(n):
        result.append(n)
        return
    d = _pollard_brent(n)
    _factor(d, result)
    _factor(n // d, result)

def _sieve(limit):
    is_prime = bytearray(b'\x01') * (limit + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_prime[i]:
            is_prime[i*i::i] = bytearray(len(is_prime[i*i::i]))
    return [i for i in range(limit + 1) if is_prime[i]]

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute primes up to 10^6
    small_primes = _sieve(1000000)
    primes_to_search = small_primes

    best_k = 0
    best_p = 3

    # Strategy: For each candidate prime p, use a segmented sieve approach.
    # For interval [p^2+1, p^2+200], we want to check if each number is p-smooth.
    # A number n in this interval is p-smooth iff after removing all prime factors <= p, remainder is 1.
    # 
    # Key insight: For p >= 20, sqrt(p^2+200) < p+1, so any composite number in [p^2+1, p^2+200]
    # that is not divisible by any prime <= p must be prime (and > p, hence not p-smooth).
    # Actually, more precisely: if n has all prime factors <= p, then n is p-smooth.
    # If n is not p-smooth, it has a prime factor > p. Since n <= p^2+200, such a factor q > p
    # implies the cofactor n/q < p + 200/p <= p+10 for p >= 20. So n = q * r where q > p is prime
    # and r < p+10.
    #
    # Alternative fast check: For each i in 1..200, val = p^2 + i.
    # val is p-smooth iff for every prime q > p, q does not divide val.
    # Equivalently: the largest prime factor of val is <= p.
    #
    # Fast method: Divide val by all primes <= p. If remainder is 1, it's p-smooth.
    # But doing this for 200 values per p is expensive.
    #
    # Better: Use the fact that if val has a prime factor q > p, then q > p and q <= p^2+200.
    # Also val/q < p + 200/p. For p >= 20, val/q <= p + 10.
    # So val = q * r where q is prime, p < q <= p^2+200, and 1 <= r <= p + 10 (for p >= 20).
    # Thus val is NOT p-smooth iff there exists a prime q in (p, p^2+200] such that q divides val
    # and val/q <= p + 10.
    #
    # Even faster: Just trial divide by primes up to p, but batch it.
    # For the segmented sieve approach:
    # Create an array rem[1..200] initialized to p^2 + i for i=1..200.
    # For each prime q <= p: for each i where rem[i] % q == 0, divide rem[i] by q (repeatedly).
    # At the end, rem[i] == 1 means p^2+i is p-smooth.
    #
    # But this is still O(200 * pi(p)) per p. For p up to 10^6, pi(p) ~ 78000, so 200*78000 = 15.6M ops per p.
    # Too slow for many p.
    #
    # Refined approach: Only need to check small primes first. Most numbers will be eliminated quickly.
    # Use early termination: as soon as we find an i where the running max prime factor exceeds p, stop.
    #
    # Actually, the key structural change: use NumPy vectorization for the small-prime division phase.
    # For a batch of candidate primes, process them together.

    # Phase 1: Exhaustive search over primes up to 10^6 with fast trial division
    # Use the observation: for p >= 20, if p^2+i has a prime factor > p, that factor q satisfies
    # p < q and p^2+i = q*r with r <= p + 10 (for p >= 20).
    # So we can check: for each i, compute p^2+i, and check if it has any prime factor in (p, p^2+200].
    # But checking all such primes is expensive.
    #
    # Simpler fast check: divide by primes up to min(p, sqrt(p^2+200)) = p for p>=20.
    # After dividing by all primes <= p, if remainder is 1, p-smooth. If remainder > 1, it's prime > p.
    #
    # The bottleneck is the trial division loop. Let's optimize by:
    # 1. Only iterating i=1,2,... until we find a non-p-smooth number (early stop)
    # 2. Using a precomputed list of small primes for the division

    for p in primes_to_search:
        if time.time() > deadline:
            break
        p_sq = p * p
        k = 0
        for i in range(1, 201):
            if time.time() > deadline:
                break
            val = p_sq + i
            t = val
            # Trial divide by primes up to p
            # Optimization: only need primes up to sqrt(t) which is ~p
            # But we can stop early if t becomes 1
            for q in small_primes:
                if q > p or q * q > t:
                    break
                while t % q == 0:
                    t //= q
            if t == 1:
                k = i
                continue
            # t > 1: t is prime (since any composite t would have factor <= sqrt(t) < p for p > 20)
            # Check if t <= p
            if t > p:
                break
            else:
                k = i
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                break

    # Phase 2: For larger primes (10^6 to 10^9), use random sampling with hybrid factorization
    if best_k < 200:
        for _ in range(5000):
            if time.time() > deadline:
                break
            log_val = 6 + random.random() * 3
            p = int(10 ** log_val)
            if p % 2 == 0:
                p += 1
            while not _is_prime_mr(p):
                p += 2
                if p > 10**9:
                    break
            if p > 10**9 or not _is_prime_mr(p):
                continue
            p_sq = p * p
            k = 0
            for i in range(1, 201):
                if time.time() > deadline:
                    break
                val = p_sq + i
                t = val
                # Divide by small primes up to 10^6
                for q in small_primes:
                    if q * q > t:
                        break
                    while t % q == 0:
                        t //= q
                if t == 1:
                    k = i
                    continue
                # t > 1: need to check if t is p-smooth
                # t could be prime or composite
                if _is_prime_mr(t):
                    if t <= p:
                        k = i
                    else:
                        break
                else:
                    factors = []
                    _factor(t, factors)
                    if all(f <= p for f in factors):
                        k = i
                    else:
                        break
            if k > best_k:
                best_k = k
                best_p = p
                if best_k == 200:
                    break

    return {'p': best_p, 'k': best_k}
