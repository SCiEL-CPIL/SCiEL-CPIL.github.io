"""Segmented-sieve approach: for each prime p, sieve the window [p^2+1, p^2+200]
with all small primes to instantly determine p-smoothness, then verify remainders."""

import time
import math
import random

# ---------- Deterministic Miller-Rabin for n < 3.3e24 ----------
_MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

def is_prime(n):
    if n < 2:
        return False
    for p in _MR_BASES:
        if n % p == 0:
            return n == p
    d = n - 1
    s = 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in _MR_BASES:
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

def pollard_rho(n):
    if n % 2 == 0:
        return 2
    if n % 3 == 0:
        return 3
    while True:
        x = random.randrange(2, n)
        y = x
        c = random.randrange(1, n)
        d = 1
        while d == 1:
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n
            d = math.gcd(abs(x - y), n)
        if d != n:
            return d

def factor(n, factors=None):
    if factors is None:
        factors = []
    if n == 1:
        return factors
    if is_prime(n):
        factors.append(n)
        return factors
    d = pollard_rho(n)
    factor(d, factors)
    factor(n // d, factors)
    return factors

def sieve_primes(limit):
    if limit < 2:
        return []
    is_p = bytearray(b'\x01') * (limit + 1)
    is_p[0] = is_p[1] = 0
    for i in range(2, int(limit**0.5) + 1):
        if is_p[i]:
            is_p[i*i::i] = bytearray(len(is_p[i*i::i]))
    return [i for i in range(2, limit + 1) if is_p[i]]

def construct():
    deadline = time.time() + 110  # safety margin before 150s kill

    # Precompute small primes for both prime generation and window sieving
    SIEVE_LIMIT = 100000
    small_primes = sieve_primes(SIEVE_LIMIT)

    best_k = 0
    best_p = 2

    # Phase 1: Search primes up to 2,000,000
    # Generate all primes in this range
    search_limit = 2000000
    all_primes = sieve_primes(search_limit)

    for p in all_primes:
        if time.time() > deadline:
            break

        base = p * p
        # We need to check p^2+1 through p^2+200
        # For each number n = p^2 + i, divide out all small primes <= p
        # After division, if remainder is 1 -> p-smooth
        # If remainder is prime -> p-smooth iff remainder <= p
        # If remainder is composite -> need to check its factors

        k = 0
        for i in range(1, 201):
            n = base + i
            v = n
            # Trial divide by small primes up to min(p, sqrt(v))
            # Optimization: only need primes up to min(p, sqrt(n))
            # For small p, sqrt(n) ~ p, so we divide by primes up to p
            # For large p, sqrt(n) ~ p as well
            limit = min(p, math.isqrt(n))

            for sp in small_primes:
                if sp > limit:
                    break
                while v % sp == 0:
                    v //= sp

            if v == 1:
                k += 1
            elif is_prime(v):
                if v <= p:
                    k += 1
                else:
                    break
            else:
                # v is composite. Check if all its prime factors are <= p.
                # Since we divided by all primes <= limit = min(p, sqrt(n)):
                # If limit == p: any remaining factor > p, not p-smooth
                # If limit == sqrt(n) < p: v has no factor <= sqrt(n), so v is prime (contradiction)
                # Actually v could be a product of two primes both > sqrt(n)? No, that's impossible.
                # If v is composite and we've removed all primes <= sqrt(v) (which is <= sqrt(n)),
                # then v must be prime. So this case shouldn't happen.
                # But to be safe, use Pollard-Brent.
                factors = factor(v)
                if all(f <= p for f in factors):
                    k += 1
                else:
                    break

        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: If we have time, search larger primes using segmented sieve
    if best_k < 200 and time.time() < deadline - 20:
        # Use segmented sieve to find primes in larger ranges
        # For each candidate prime p, do the window check
        # Key insight: for large p (> 10^6), the window [p^2+1, p^2+200] has numbers ~p^2
        # After dividing by primes <= p, the remainder is either 1 or a prime > p (since
        # any composite remainder would need a factor <= sqrt(remainder) < p, which we divided out)
        # So for large p, we just need to check if the remainder after small-prime division is 1 or a prime <= p

        # Segmented sieve for primes in [2M, 50M]
        segment_start = 2000000
        segment_end = 50000000
        seg_size = 1000000

        pos = segment_start
        while pos < segment_end and time.time() < deadline - 10:
            seg_end = min(pos + seg_size, segment_end)
            # Segmented sieve
            is_prime_seg = bytearray(b'\x01') * (seg_end - pos)

            for sp in small_primes:
                if sp * sp > seg_end:
                    break
                first = ((pos + sp - 1) // sp) * sp
                if first == sp:
                    first += sp
                for j in range(first - pos, seg_end - pos, sp):
                    is_prime_seg[j] = 0

            # Check each prime in this segment
            for i in range(seg_end - pos):
                if time.time() > deadline - 5:
                    break
                cand = pos + i
                if cand < 2 or not is_prime_seg[i]:
                    continue
                # Quick primality check (segmented sieve might have false positives for small composites)
                if not is_prime(cand):
                    continue

                p = cand
                base = p * p
                k = 0
                for ii in range(1, 201):
                    n = base + ii
                    v = n
                    limit = min(p, math.isqrt(n))

                    for sp in small_primes:
                        if sp > limit:
                            break
                        while v % sp == 0:
                            v //= sp

                    if v == 1:
                        k += 1
                    elif is_prime(v):
                        if v <= p:
                            k += 1
                        else:
                            break
                    else:
                        factors = factor(v)
                        if all(f <= p for f in factors):
                            k += 1
                        else:
                            break

                if k > best_k:
                    best_k = k
                    best_p = p

            pos = seg_end

    if best_p == 0:
        best_p = 2
        best_k = 1

    return {'p': best_p, 'k': best_k}
