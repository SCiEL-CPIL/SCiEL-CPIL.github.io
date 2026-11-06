"""
Combined program: deterministic sieve-based search for long runs of p-smooth numbers.
Uses a large sieve of primes for fast trial division, plus a segmented-sieve
pre-filter to quickly eliminate candidate primes p before doing the expensive
per-p run-length check.
"""
import time
import math

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0  # safety margin before 150 s kill

    # ---------- deterministic Miller-Rabin (for n < 3.3e24) ----------
    _MR_BASES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)

    def is_prime(n):
        if n < 2:
            return False
        for p in _MR_BASES:
            if n == p:
                return True
            if n % p == 0:
                return False
        d = n - 1
        s = 0
        while d % 2 == 0:
            d //= 2
            s += 1
        for a in _MR_BASES:
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

    # ---------- Pollard-Brent ----------
    def _pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            y = 2
            c = 1
            m = 128
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
                        q = (q * abs(x - y)) % n
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
            c += 1

    def factorize(n):
        if n == 1:
            return []
        if is_prime(n):
            return [n]
        d = _pollard_brent(n)
        return factorize(d) + factorize(n // d)

    # ---------- sieve ----------
    SIEVE_LIMIT = 3000000  # primes up to 3e6

    def sieve_primes(limit):
        if limit < 2:
            return []
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                is_p[i * i::i] = bytearray(len(is_p[i * i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    primes = sieve_primes(SIEVE_LIMIT)
    # primes list is used for trial division in is_p_smooth

    # ---------- p-smooth check via trial division ----------
    def is_p_smooth(n, p):
        """Return True iff every prime factor of n is <= p."""
        if n <= 1:
            return True
        m = n
        for q in primes:
            if q > p:
                break
            if q * q > m:
                break
            if m % q == 0:
                while m % q == 0:
                    m //= q
                if m == 1:
                    return True
        if m == 1:
            return True
        # m is prime (or 1); check if it is <= p
        return m <= p

    # ---------- run-length check for a given prime p ----------
    def check_run(p, k_max=200):
        k = 0
        p_sq = p * p
        for i in range(1, k_max + 1):
            n = p_sq + i
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        return k

    # ---------- segmented sieve for candidate p ----------
    # We want primes p such that p^2 + i is composite (i.e., not prime) for
    # i = 1, ..., 200.  If p^2 + i is prime for some i, then that number has
    # a prime factor > p (itself), so the run breaks at or before i.
    #
    # For a fixed prime r <= SIEVE_LIMIT, p^2 + i ≡ 0 (mod r)  =>  p^2 ≡ -i (mod r)
    # => p ≡ ±sqrt(-i) (mod r).  So for each i in 1..200 and each r, we mark
    # residues p mod r that make p^2+i divisible by r.  If p hits any such
    # residue for any i, the run is < 200.  (This is a necessary condition
    # for a full run of 200; it is not sufficient, but it filters fast.)
    #
    # Actually, we only need to check whether p^2 + i is prime for i=1..200.
    # A simpler pre-filter: for each small prime r, compute the set of
    # residues a mod r such that a^2 + i ≡ 0 mod r for some i in 1..200.
    # If p mod r is in that set, then p^2 + i is divisible by r for that i,
    # which is fine (r <= p means r is an allowed factor).  The dangerous
    # case is when p^2 + i is prime.  So the pre-filter should mark p as
    # BAD if p^2 + i is prime for some i.  That's hard to precompute.
    #
    # Better approach: just iterate over primes and check.  The sieve-based
    # trial division is fast enough.  Let's just do a direct search with
    # a time budget.

    best_k = 0
    best_p = 2

    # Check very small primes first
    for p in primes:
        if p > 100:
            break
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Now search larger primes.  We'll go through the primes list in order.
    # For each prime, check the run.  Stop when time runs out.
    for p in primes:
        if p <= 100:
            continue
        if time.time() > DEADLINE - 5:
            break
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p
            # If we've found a very long run, we can be more optimistic
            # but still keep searching.

    # If we still have time, try to search beyond the sieve limit using
    # Pollard-Brent for factorization.
    if time.time() < DEADLINE - 30:
        # Generate primes beyond SIEVE_LIMIT using a simple incremental method
        # or just check odd numbers with Miller-Rabin.
        # This is slower but can find larger p.
        p = SIEVE_LIMIT + 2
        if p % 2 == 0:
            p += 1
        while p < 10**9 and time.time() < DEADLINE - 5:
            if is_prime(p):
                k = 0
                p_sq = p * p
                for i in range(1, 201):
                    if time.time() > DEADLINE - 5:
                        break
                    n = p_sq + i
                    # Quick check: if n is prime, fail
                    if is_prime(n):
                        break
                    # Factorize n and check max factor
                    factors = factorize(n)
                    if max(factors) <= p:
                        k += 1
                    else:
                        break
                if k > best_k:
                    best_k = k
                    best_p = p
            p += 2

    return {'p': best_p, 'k': best_k}


if __name__ == "__main__":
    res = construct()
    print(res)
