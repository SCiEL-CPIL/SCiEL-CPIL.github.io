import math
import time
import random

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]

def is_prime(n):
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _SMALL_PRIMES:
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

def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    if n % 5 == 0:
        return 5
    x = 2
    y = 2
    d = 1
    c = 1
    while d == 1:
        x = (x * x + c) % n
        y = (y * y + c) % n
        y = (y * y + c) % n
        d = math.gcd(abs(x - y), n)
        if d == n:
            c += 1
            x = 2
            y = 2
            d = 1
            if c > 10:
                x = 3
                y = 3
                c = 1
    return d

def largest_prime_factor(n):
    if n == 1:
        return 1
    if n < 0:
        n = -n
    for p in _SMALL_PRIMES:
        if n % p == 0:
            while n % p == 0:
                n //= p
            if n == 1:
                return max(_SMALL_PRIMES[:len(_SMALL_PRIMES)])
    stack = [n]
    max_f = 1
    while stack:
        m = stack.pop()
        if m == 1:
            continue
        if is_prime(m):
            if m > max_f:
                max_f = m
        else:
            d = pollard_rho(m)
            stack.append(d)
            stack.append(m // d)
    return max_f

def _sieve_primes(limit):
    if limit < 2:
        return []
    sieve = bytearray(b'\x01') * (limit + 1)
    sieve[0] = False
    sieve[1] = False
    for i in range(2, int(limit ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, limit + 1) if sieve[i]]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140.0  # safety margin before 150s limit

    # Precompute primes up to 10^7 for candidate p's
    SIEVE_LIMIT = 10_000_000
    primes = _sieve_primes(SIEVE_LIMIT)

    best_k = 0
    best_p = 2

    # Check run length for a given prime p
    def check_run(p, k_max=200):
        k = 0
        p_sq = p * p
        for i in range(1, k_max + 1):
            if time.time() > DEADLINE - 3:
                break
            n = p_sq + i
            if is_prime(n):
                if n > p:
                    break
                else:
                    k += 1
                    continue
            lf = largest_prime_factor(n)
            if lf > p:
                break
            k += 1
        return k

    # Search primes in increasing order
    for p in primes:
        if time.time() > DEADLINE - 10:
            break
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    # If we have time, try some larger primes beyond the sieve
    if time.time() < DEADLINE - 20:
        random.seed(42)
        for _ in range(50):
            if time.time() > DEADLINE:
                break
            p_candidate = random.randint(10_000_000, 1_000_000_000)
            if p_candidate % 2 == 0:
                p_candidate += 1
            if is_prime(p_candidate):
                p = p_candidate
                k = check_run(p)
                if k > best_k:
                    best_k = k
                    best_p = p

    return {'p': best_p, 'k': best_k}
