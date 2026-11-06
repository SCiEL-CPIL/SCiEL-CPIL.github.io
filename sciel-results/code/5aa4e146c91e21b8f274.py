import time
import math
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill

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
            y = random.randrange(1, n)
            c = random.randrange(1, n)
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

    def factorize(n):
        if n == 1:
            return []
        if is_prime(n):
            return [n]
        d = _pollard_brent(n)
        return factorize(d) + factorize(n // d)

    # ---------- sieve ----------
    def sieve_primes(limit):
        if limit < 2:
            return []
        is_p = bytearray(b'\x01') * (limit + 1)
        is_p[0] = is_p[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if is_p[i]:
                is_p[i * i::i] = bytearray(len(is_p[i * i::i]))
        return [i for i in range(2, limit + 1) if is_p[i]]

    # ---------- p-smooth check via trial division + Pollard-Brent ----------
    SIEVE_LIMIT = 100000
    primes_for_division = sieve_primes(SIEVE_LIMIT)

    def is_p_smooth(n, p):
        """Return True iff every prime factor of n is <= p."""
        if n <= 1:
            return True
        m = n
        for q in primes_for_division:
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
        if is_prime(m):
            return m <= p
        # m is composite; factorize it
        factors = factorize(m)
        return all(f <= p for f in factors)

    # ---------- run-length check for a given prime p ----------
    def check_run(p, k_max=200):
        k = 0
        p_sq = p * p
        for i in range(1, k_max + 1):
            if time.time() > deadline:
                break
            n = p_sq + i
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        return k

    # ---------- Precomputation: forbidden residues for small primes q ----------
    # For each prime q and each i in [1, 200], compute residues r mod q where r^2 ≡ -i (mod q)
    # If p ≡ r (mod q) and q > p, then p^2 + i ≡ 0 (mod q) means q divides p^2+i,
    # so p^2+i has a prime factor > p.
    
    # We'll precompute for primes q up to Q_PRECOMP_LIMIT
    Q_PRECOMP_LIMIT = 10000
    small_primes_q = sieve_primes(Q_PRECOMP_LIMIT)
    
    # q_roots[q][i] = list of residues r in [0, q-1] such that r^2 ≡ -i (mod q)
    q_roots = {}
    for q in small_primes_q:
        q_roots[q] = {}
        for i in range(1, 201):
            target = (-i) % q
            roots = []
            if target == 0:
                roots = [0]
            elif q == 2:
                if target == 1:
                    roots = [1]
            else:
                # Check if target is a quadratic residue mod q using Euler's criterion
                if pow(target, (q - 1) // 2, q) == 1:
                    # Find sqrt using Tonelli-Shanks
                    def tonelli_shanks(n, p):
                        if p % 4 == 3:
                            return pow(n, (p + 1) // 4, p)
                        q_val = p - 1
                        s = 0
                        while q_val % 2 == 0:
                            q_val //= 2
                            s += 1
                        z = 2
                        while pow(z, (p - 1) // 2, p) != p - 1:
                            z += 1
                        m = s
                        c = pow(z, q_val, p)
                        t = pow(n, q_val, p)
                        r = pow(n, (q_val + 1) // 2, p)
                        while t != 1:
                            i_val = 0
                            t2 = t
                            while t2 != 1:
                                t2 = (t2 * t2) % p
                                i_val += 1
                            b = pow(c, 1 << (m - i_val - 1), p)
                            m = i_val
                            c = (b * b) % p
                            t = (t * c) % p
                            r = (r * b) % p
                        return r
                    root1 = tonelli_shanks(target, q)
                    root2 = (q - root1) % q
                    roots = [root1, root2] if root1 != root2 else [root1]
            q_roots[q][i] = roots

    # ---------- Quick check using precomputed residues ----------
    def quick_check(p, max_i=200):
        """Return the maximum k such that for all i in [1, k], p^2+i is not divisible by any prime q in (p, Q_PRECOMP_LIMIT]."""
        k = 0
        for i in range(1, max_i + 1):
            bad = False
            for q in small_primes_q:
                if q <= p:
                    continue
                if q > Q_PRECOMP_LIMIT:
                    break
                p_mod_q = p % q
                if p_mod_q in q_roots[q][i]:
                    bad = True
                    break
            if bad:
                break
            k += 1
        return k

    # ---------- Main search ----------
    best_k = 0
    best_p = 2

    # Generate candidate primes up to SIEVE_LIMIT
    SEARCH_SIEVE_LIMIT = 5000000
    candidate_primes = sieve_primes(SEARCH_SIEVE_LIMIT)

    # Check small primes first
    for p in candidate_primes:
        if p > 100:
            break
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Now search larger primes with quick check filter
    for p in candidate_primes:
        if p <= 100:
            continue
        if time.time() > deadline - 5:
            break
        
        # Quick check to filter out most candidates
        k_quick = quick_check(p)
        
        # Only do full check if quick check is promising
        if k_quick > best_k or (k_quick >= best_k - 3 and k_quick > 0):
            k_full = check_run(p)
            if k_full > best_k:
                best_k = k_full
                best_p = p

    # If time remains, try some larger primes using Miller-Rabin
    if time.time() < deadline - 20:
        random.seed(42)
        for _ in range(200):
            if time.time() > deadline - 5:
                break
            p = random.randrange(10**7, 10**9)
            p |= 1  # Make it odd
            if not is_prime(p):
                continue
            k_quick = quick_check(p)
            if k_quick > best_k or (k_quick >= best_k - 3 and k_quick > 0):
                k_full = check_run(p)
                if k_full > best_k:
                    best_k = k_full
                    best_p = p

    return {'p': best_p, 'k': best_k}
