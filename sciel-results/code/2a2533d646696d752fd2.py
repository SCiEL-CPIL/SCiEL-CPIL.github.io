import time
import math
import random

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

# ---------- Sieve ----------
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

    # Precompute small primes up to 100000 for fast trial division
    small_primes = _sieve(100000)
    primes_to_search = _sieve(1000000)  # search primes up to 10^6 deterministically

    best_k = 0
    best_p = 3

    for p in primes_to_search:
        if time.time() > deadline:
            break
        if p > 10**9:
            break
        p_sq = p * p
        k = 0
        for i in range(1, 201):
            if time.time() > deadline:
                break
            val = p_sq + i
            # Quick check: divide by small primes
            t = val
            has_large_factor = False
            for q in small_primes:
                if q > p:
                    break
                if q * q > t:
                    break
                while t % q == 0:
                    t //= q
            if t == 1:
                k = i
                continue
            # t > 1: check if t is prime and <= p
            if _is_prime_mr(t):
                if t <= p:
                    k = i
                else:
                    has_large_factor = True
            else:
                # Need full factorization
                factors = []
                _factor(t, factors)
                if all(f <= p for f in factors):
                    k = i
                else:
                    has_large_factor = True
            if has_large_factor:
                break
        if k > best_k:
            best_k = k
            best_p = p
            if best_k == 200:
                break

    # If we haven't found k=200, try some larger primes with random sampling
    if best_k < 200:
        for _ in range(5000):
            if time.time() > deadline:
                break
            # Sample primes in [10^6, 10^9]
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
                has_large_factor = False
                for q in small_primes:
                    if q > p:
                        break
                    if q * q > t:
                        break
                    while t % q == 0:
                        t //= q
                if t == 1:
                    k = i
                    continue
                if _is_prime_mr(t):
                    if t <= p:
                        k = i
                    else:
                        has_large_factor = True
                else:
                    factors = []
                    _factor(t, factors)
                    if all(f <= p for f in factors):
                        k = i
                    else:
                        has_large_factor = True
                if has_large_factor:
                    break
            if k > best_k:
                best_k = k
                best_p = p
                if best_k == 200:
                    break

    return {'p': best_p, 'k': best_k}
