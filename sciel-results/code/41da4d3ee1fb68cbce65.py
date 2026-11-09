"""Rewritten program: bitset-sieve over primes p to find long p-smooth runs after p^2."""
# EVOLVE-BLOCK-START
import time
import math
import random
import sys

def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    MAX_K = 200

    # ---------- Deterministic Miller-Rabin for n < 2^64 ----------
    def is_prime_mr(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return n in (2, 3)
        d = n - 1
        r = 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for a in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if a >= n:
                continue
            x = pow(a, d, n)
            if x == 1 or x == n - 1:
                continue
            for _ in range(r - 1):
                x = x * x % n
                if x == n - 1:
                    break
            else:
                return False
        return True

    # ---------- Pollard Brent ----------
    def pollard_brent(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        if n % 5 == 0:
            return 5
        while True:
            y = random.randint(2, n - 1)
            c = random.randint(1, n - 1)
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
            if 1 < g < n:
                return g

    # ---------- Full factorization ----------
    def factorize(n):
        if n <= 1:
            return []
        if is_prime_mr(n):
            return [n]
        d = pollard_brent(n)
        return factorize(d) + factorize(n // d)

    # ---------- Sieve primes up to limit ----------
    def sieve_primes(limit):
        if limit < 2:
            return []
        s = bytearray(b'\x01') * (limit + 1)
        s[0] = s[1] = 0
        for i in range(2, int(limit ** 0.5) + 1):
            if s[i]:
                step = i
                start = i * i
                s[start::step] = bytearray(len(s[start::step]))
        return [i for i in range(2, limit + 1) if s[i]]

    # ---------- Precompute small primes for trial division ----------
    SMALL_LIMIT = 100000
    small_primes = sieve_primes(SMALL_LIMIT)

    # ---------- Check if n is p-smooth (all prime factors <= p) ----------
    def is_p_smooth(n, p):
        if n <= 1:
            return True
        # Trial divide by small primes <= p
        for sp in small_primes:
            if sp > p:
                break
            while n % sp == 0:
                n //= sp
        if n == 1:
            return True
        if n > p:
            # Need full factorization
            factors = factorize(n)
            return all(f <= p for f in factors)
        else:
            # n <= p, so all its prime factors are <= n <= p
            return True

    # ---------- Quick smooth run check (fast, conservative) ----------
    def quick_smooth_run(p, max_k=200):
        """Returns the length of the run of p-smooth numbers starting at p^2+1.
        Uses only small primes; if remainder > p and composite, conservatively stops."""
        k = 0
        p_sq = p * p
        for i in range(1, max_k + 1):
            n = p_sq + i
            if n <= 1:
                continue
            # Trial divide by small primes <= p
            for sp in small_primes:
                if sp > p:
                    break
                if sp * sp > n:
                    break
                while n % sp == 0:
                    n //= sp
            if n == 1:
                k += 1
            elif n <= p:
                k += 1
            else:
                # n > p: check if prime (fail) or composite
                if is_prime_mr(n):
                    break
                else:
                    # Composite > p: need to verify all factors <= p
                    # For quick check, assume fail to be safe (conservative)
                    break
        return k

    # ---------- Full verification ----------
    def verify_run(p, k):
        p_sq = p * p
        for i in range(1, k + 1):
            n = p_sq + i
            if not is_p_smooth(n, p):
                return False
        return True

    best_k = 0
    best_p = 2

    # ---------- Phase 1: Dense search over small primes ----------
    # For small p, p^2 is small so smoothness is more likely
    small_search_limit = 200000
    primes_small = sieve_primes(small_search_limit)

    for p in primes_small:
        if time.time() > deadline:
            break
        k = quick_smooth_run(p, MAX_K)
        if k > best_k:
            if verify_run(p, k):
                best_k = k
                best_p = p

    # ---------- Phase 2: Chunked sieve for medium primes ----------
    # Search primes in [200000, 5_000_000] in chunks
    CHUNK_SIZE = 100000
    start_chunk = small_search_limit
    end_chunk = min(5_000_000, 10_000_000)

    for chunk_start in range(start_chunk, end_chunk, CHUNK_SIZE):
        if time.time() > deadline:
            break
        chunk_end = min(chunk_start + CHUNK_SIZE, end_chunk)
        # Sieve this chunk
        s = bytearray(b'\x01') * (chunk_end - chunk_start)
        # Mark composites using primes up to sqrt(chunk_end)
        for sp in small_primes:
            if sp * sp > chunk_end:
                break
            # Find first multiple of sp in [chunk_start, chunk_end)
            first = ((chunk_start + sp - 1) // sp) * sp
            if first < sp * sp:
                first = sp * sp
            for j in range(first - chunk_start, chunk_end - chunk_start, sp):
                s[j] = 0
        # Collect primes in this chunk
        for offset in range(0, len(s)):
            val = chunk_start + offset
            if val < 2:
                continue
            if s[offset] and is_prime_mr(val):
                if time.time() > deadline:
                    break
                k = quick_smooth_run(val, MAX_K)
                if k > best_k:
                    if verify_run(val, k):
                        best_k = k
                        best_p = val

    # ---------- Phase 3: Targeted search for larger primes ----------
    # For larger p, use a smarter approach:
    # p^2 + i is p-smooth means all prime factors of p^2+i are <= p.
    # Key insight: if q > p is prime and q | (p^2 + i), then p^2 ≡ -i (mod q).
    # For each offset i in [1, 200], the "bad" primes q > p that divide p^2+i
    # are those where -i is a quadratic residue mod q.
    # We can precompute for small i which primes q make p^2+i have a factor > p.

    # Alternative: for larger primes, do random sampling with verification
    if time.time() < deadline:
        # Try structured ranges where smooth numbers are denser
        # Focus on primes p where p^2 is near a highly composite number
        for attempt in range(2000):
            if time.time() > deadline:
                break
            # Random prime in various ranges
            r = random.random()
            if r < 0.3:
                lo, hi = 5_000_000, 50_000_000
            elif r < 0.6:
                lo, hi = 50_000_000, 200_000_000
            elif r < 0.85:
                lo, hi = 200_000_000, 500_000_000
            else:
                lo, hi = 500_000_000, 1_000_000_000

            candidate = random.randint(lo, hi)
            if candidate % 2 == 0:
                candidate += 1

            # Find next prime
            found_prime = False
            for _ in range(200):
                if is_prime_mr(candidate):
                    found_prime = True
                    break
                candidate += 2
            if not found_prime:
                continue

            # Quick check
            k = quick_smooth_run(candidate, MAX_K)
            if k > best_k:
                if verify_run(candidate, k):
                    best_k = k
                    best_p = candidate

    # ---------- Phase 4: Exhaustive check of very small primes for max k ----------
    # Very small primes (p=2,3,5,7,11,13,17,19,23,29,31,37,41,43,47) may give long runs
    # because p^2 is small and the numbers p^2+1...p^2+k are small
    for p in [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if time.time() > deadline:
            break
        # Do full verification up to 200
        p_sq = p * p
        k = 0
        for i in range(1, MAX_K + 1):
            n = p_sq + i
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        if k > best_k:
            best_k = k
            best_p = p

    # ---------- Phase 5: If time remains, do a denser sweep of medium primes ----------
    if time.time() < deadline - 5:
        # Sweep primes in [10000, 200000] more carefully (already done, but recheck top candidates)
        # Instead, try primes near powers of 2 or other structured values
        for base in [1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000, 500000, 1000000, 2000000, 5000000, 10000000, 20000000, 50000000, 100000000, 200000000, 500000000]:
            if time.time() > deadline:
                break
            # Find primes near base
            candidate = base
            if candidate % 2 == 0:
                candidate += 1
            for _ in range(50):
                if is_prime_mr(candidate):
                    k = quick_smooth_run(candidate, MAX_K)
                    if k > best_k:
                        if verify_run(candidate, k):
                            best_k = k
                            best_p = candidate
                candidate += 2

    return {'p': best_p, 'k': best_k}
# EVOLVE-BLOCK-END
