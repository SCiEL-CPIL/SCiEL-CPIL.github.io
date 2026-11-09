"""Rewritten program: fast factoring with Miller-Rabin + Pollard-Brent, multi-phase prime search."""
# EVOLVE-BLOCK-START
import time
import random
import math

def construct():
    t0 = time.time()
    DEADLINE = t0 + 140.0  # 10s safety margin before 150s kill

    # ---------- Deterministic Miller-Rabin for n < 3,317,044,064,679,887,385,961,981 ----------
    # For our purposes (n up to ~10^18), these bases are sufficient:
    MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

    def is_prime(n):
        if n < 2:
            return False
        for p in MR_BASES:
            if n == p:
                return True
            if n % p == 0:
                return False
        # Write n-1 as 2^r * d
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in MR_BASES:
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

    # ---------- Pollard's Brent's algorithm ----------
    def gcd(a, b):
        while b:
            a, b = b, a % b
        return a

    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = random.randrange(2, n)
            c = random.randrange(1, n)
            m = 128
            g = r = q = 1
            x = ys = y
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
                    g = gcd(q, n)
                    k += m
                r *= 2
            if g == n:
                g = 1
                while g == 1:
                    ys = (ys * ys + c) % n
                    g = gcd(abs(x - ys), n)
            if g != n:
                return g

    # ---------- Factor n completely, return list of prime factors ----------
    def factor(n):
        if n <= 1:
            return []
        if is_prime(n):
            return [n]
        # Try small primes first
        for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]:
            if n % p == 0:
                return [p] + factor(n // p)
        d = pollard_brent(n)
        return factor(d) + factor(n // d)

    # ---------- Check if n is p-smooth (all prime factors <= p) ----------
    def is_p_smooth(n, p):
        if n <= 1:
            return True
        if n <= p:
            return is_prime(n) or True  # if n<=p, its prime factors are <= n <= p
        # Factor n
        factors = factor(n)
        return all(f <= p for f in factors)

    # ---------- Sieve for small primes ----------
    def sieve_primes(limit):
        if limit < 2:
            return []
        is_p = [True] * (limit + 1)
        is_p[0] = is_p[1] = False
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                for j in range(i * i, limit + 1, i):
                    is_p[j] = False
        return [i for i in range(2, limit + 1) if is_p[i]]

    # ---------- Evaluate a prime p: find max k such that p^2+1...p^2+k are p-smooth ----------
    def evaluate_p(p):
        k = 0
        while k < 200:
            n = p * p + k + 1
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        return k

    # ---------- Phase 1: Exhaustive search over small primes (up to ~50000) ----------
    # Use sieve to get primes, then evaluate each
    small_limit = 50000
    small_primes = sieve_primes(small_limit)
    best_k = 0
    best_p = 2

    for p in small_primes:
        if time.time() > DEADLINE:
            break
        k = evaluate_p(p)
        if k > best_k:
            best_k = k
            best_p = p

    # ---------- Phase 2: Medium primes (50000 to ~200000) with systematic sampling ----------
    # Check primes in this range, but sample to stay within time
    med_limit = 200000
    med_primes = sieve_primes(med_limit)
    # Filter to only those > small_limit
    med_primes = [p for p in med_primes if p > small_limit]

    # Sample: check every prime but with a time guard
    step = max(1, len(med_primes) // 2000) if time.time() < DEADLINE - 60 else 1
    for i in range(0, len(med_primes), step):
        if time.time() > DEADLINE:
            break
        p = med_primes[i]
        k = evaluate_p(p)
        if k > best_k:
            best_k = k
            best_p = p

    # ---------- Phase 3: Large primes (up to 10^9) with random sampling ----------
    # For large primes, we can't sieve. Use random odd numbers and test primality.
    # The key insight: for very large p, p^2+1 is ~10^18, and we need all factors <= p.
    # This is harder to satisfy, but we sample randomly.
    random.seed(42)
    large_start = med_limit + 1
    large_end = 10**9

    # How many random primes to try
    attempts = 0
    max_attempts = 5000

    while time.time() < DEADLINE - 5 and attempts < max_attempts:
        # Generate a random odd number in [large_start, large_end]
        if large_end - large_start < 10:
            break
        candidate = random.randrange(large_start, large_end)
        if candidate % 2 == 0:
            candidate += 1
        # Ensure it's odd
        if candidate < 3:
            continue
        if not is_prime(candidate):
            attempts += 1
            continue
        attempts += 1
        k = evaluate_p(candidate)
        if k > best_k:
            best_k = k
            best_p = candidate

    # ---------- Also try some structured large primes ----------
    # Primes near powers of 2 or other structured locations might have special properties
    # Try primes of form 2^m * a + b for various m
    if time.time() < DEADLINE - 5:
        for m in range(16, 30):
            base = 1 << m
            for offset in range(-100, 101, 2):
                if time.time() > DEADLINE - 5:
                    break
                candidate = base + offset
                if candidate < 3 or candidate > 10**9:
                    continue
                if candidate % 2 == 0:
                    continue
                if is_prime(candidate):
                    k = evaluate_p(candidate)
                    if k > best_k:
                        best_k = k
                        best_p = candidate

    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
