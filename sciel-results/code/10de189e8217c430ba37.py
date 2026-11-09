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

    # Precompute primes up to 2*10^6 for trial division
    SIEVE_LIMIT = 2_000_000
    small_primes = _sieve(SIEVE_LIMIT)
    small_primes_arr = np.array(small_primes, dtype=np.int64)

    best_k = 0
    best_p = 3

    # ---- Phase 1: Exhaustive vectorized search over primes up to ~5*10^6 ----
    # For a batch of B candidate primes, we build a B x 200 array of values p^2+i,
    # then for each small prime q, we check divisibility across all 200 values
    # for all B primes simultaneously using NumPy.
    #
    # Key optimization: we only need to strip primes q <= p for each p.
    # Since p is at most ~5*10^6, we need primes up to 5*10^6.
    # But we can limit: for a given p, any prime factor q > sqrt(p^2+200) ≈ p+100/p
    # would make the cofactor < p, so if after stripping primes up to p, remainder > 1,
    # it must be prime. We check if that prime <= p.
    #
    # Vectorized approach:
    #   vals = p_batch[:, None] ** 2 + np.arange(1, 201)[None, :]   # shape (B, 200)
    #   For each q in small_primes:
    #       mask = vals % q == 0
    #       while any(mask): vals[mask] //= q; mask = vals[mask] % q == 0
    #   At end: vals == 1 means p-smooth. Otherwise vals is a prime (or product of two large primes).
    #   Check: if vals <= p for that row, it's p-smooth; else not.
    #
    # But full vectorization over 200 columns for each q is expensive.
    # Better: process in chunks and use early termination per row.

    # Precompute offsets
    offsets = np.arange(1, 201, dtype=np.int64)  # shape (200,)

    # We'll process primes in batches. For each batch, we do the sieve.
    # Batch size: choose so that B * 200 * len(primes_to_strip) is manageable.
    # For p up to 5*10^6, pi(p) ~ 350,000. That's too many for full vectorization.
    # Instead: only strip primes up to a threshold (e.g., 10^5), then handle the rest with Pollard-Brent.
    # Actually, for p up to 5*10^6, sqrt(p^2+200) ≈ p. So we need to strip primes up to p.
    # But most numbers will be reduced to 1 or a small prime after stripping the first few primes.
    #
    # Revised strategy: For each batch, strip primes up to 10^4 (1229 primes) vectorized.
    # Then for remaining values > 1, check if they're prime and <= p.
    # If a value is composite after stripping primes up to 10^4, it has a factor in (10^4, p].
    # For p <= 5*10^6, that's at most one such factor (since (10^4)^2 = 10^8 > p^2 for p < 10^4,
    # wait no: for p > 10^4, (10^4)^2 = 10^8 < p^2, so there could be two factors in (10^4, p]).
    # Hmm, this is getting complicated. Let me just strip all primes up to p but do it smartly.
    #
    # Actually, the most efficient approach for the vectorized case:
    # For each candidate p, the values p^2+1 ... p^2+200.
    # A value v = p^2 + i. If v has a prime factor q > p, then q > p and v/q < p + 200/p.
    # For p >= 100, v/q < p + 2. So v = q * r where q is prime, p < q, and 1 <= r <= p+2.
    # So v is NOT p-smooth iff there exists a prime q in (p, p^2+200] such that q | v and v/q <= p+2.
    #
    # Even simpler check: after dividing by all primes <= p, if remainder is 1, p-smooth.
    # The vectorized sieve does exactly this.
    #
    # Let's limit the vectorized search to primes up to 200,000 where pi(p) ~ 17984.
    # For a batch of B=50 primes, the array is 50 x 200 = 10000 elements.
    # For each of 17984 primes, we do a modulo and conditional division on 10000 elements.
    # That's 17984 * 10000 = 179.8M operations per batch. Too slow.
    #
    # Better: only strip primes up to sqrt(max_value) ≈ p for the relevant p.
    # But actually, we can be smarter: for the vectorized sieve, we only need to strip
    # primes q where q <= p AND q^2 <= p^2+200, which is always true for q <= p.
    # The issue is the number of primes to strip.
    #
    # New plan: Use a two-phase approach.
    # Phase 1a: Vectorized strip of primes up to 10,000 (1229 primes). This is fast.
    #   After this, any remaining factor > 10000 in v = p^2+i must be prime (since (10000)^2 = 10^8,
    #   and for p > 10000, p^2 > 10^8, so v could have two factors > 10000... no wait,
    #   if v has two factors both > 10000, their product > 10^8. For p > 10000, p^2 > 10^8,
    #   so v = p^2+i could indeed have two factors > 10000. But for p <= 10000, v <= 10^8+200,
    #   and two factors > 10000 would have product > 10^8, so at most one factor > 10000.
    #
    # For p > 10000: after stripping primes <= 10000, the remainder r of v is either 1,
    # a prime, or a product of two primes both > 10000. In the latter case, both are > 10000
    # and their product = r <= v <= p^2+200. Since both > 10000, we need to check if both <= p.
    #
    # This is getting complex. Let me just do a simpler but effective approach:
    # For each candidate p, do a fast check using only the first ~100 small primes (up to ~541).
    # If all 200 values become 1 after this, great. Otherwise, do full factorization on the
    # specific values that didn't reduce to 1.
    #
    # Actually, the BEST structural change: instead of searching over p, search over the
    # "gap structure". For a given p, the condition is that p^2+1, ..., p^2+k are all p-smooth.
    # This is equivalent to: for each i in 1..k, all prime factors of p^2+i are <= p.
    #
    # Let me just implement a clean vectorized version for small p and Pollard-Brent for large p.

    # Phase 1: Primes up to 50,000 - vectorized small-prime stripping
    phase1_primes = [p for p in small_primes if p <= 50000]
    
    # Strip primes up to 10,000 vectorized
    strip_primes = [p for p in small_primes if p <= 10000]
    strip_primes_arr = np.array(strip_primes, dtype=np.int64)
    
    BATCH = 20
    for batch_start in range(0, len(phase1_primes), BATCH):
        if time.time() > deadline:
            break
        batch = phase1_primes[batch_start:batch_start+BATCH]
        B = len(batch)
        p_arr = np.array(batch, dtype=np.int64)
        
        # Build the (B, 200) array
        vals = p_arr[:, None] * p_arr[:, None] + offsets[None, :]
        
        # Strip each prime in strip_primes
        for q in strip_primes:
            # For rows where p >= q, strip q from vals
            mask_p = p_arr >= q
            if not np.any(mask_p):
                continue
            # Only process rows where p >= q
            sub_vals = vals[mask_p]
            # Strip q
            for _ in range(8):  # max exponent
                div_mask = (sub_vals % q == 0)
                if not np.any(div_mask):
                    break
                sub_vals[div_mask] //= q
            vals[mask_p] = sub_vals
        
        # Now check: for each row i, vals[i,j] should be 1 or a prime <= p[i]
        # If vals[i,j] > 1 and (not prime or > p[i]), then it's not p-smooth
        k_counts = np.zeros(B, dtype=np.int32)
        for i in range(B):
            p_val = batch[i]
            row = vals[i]
            k = 0
            for j in range(200):
                v = row[j]
                if v == 1:
                    k = j + 1
                    continue
                # v > 1: check if it's prime and <= p_val
                if v <= p_val:
                    # Could be prime <= p_val (good) or composite with all factors <= p_val (good)
                    # Since we stripped primes up to 10000, if v <= 10000, it's 1 (already handled)
                    # If 10000 < v <= p_val, v is prime (since any composite v would have a factor <= sqrt(v) < 10000 for v < 10^8)
                    # Wait: sqrt(10^8) = 10000. So if v < 10^8 and composite, it has a factor <= 10000, which we stripped.
                    # So v > 1 after stripping means v is prime.
                    # If v <= p_val, it's p-smooth.
                    k = j + 1
                else:
                    # v > p_val: v is prime (as argued) and > p_val, so NOT p-smooth
                    break
            k_counts[i] = k
        
        for i in range(B):
            if k_counts[i] > best_k:
                best_k = int(k_counts[i])
                best_p = batch[i]
                if best_k == 200:
                    return {'p': best_p, 'k': best_k}
    
    # Phase 2: Primes from 50,000 to 5,000,000 - hybrid approach
    # For each p, do quick filter with small primes, then Pollard-Brent on survivors
    phase2_primes = [p for p in small_primes if 50000 < p <= 5_000_000]
    
    # Small primes for quick filter (up to 1000)
    quick_primes = [p for p in small_primes if p <= 1000]
    
    for p in phase2_primes:
        if time.time() > deadline:
            break
        p_sq = p * p
        k = 0
        for i in range(1, 201):
            if time.time() > deadline:
                break
            val = p_sq + i
            t = val
            # Quick strip with small primes
            for q in quick_primes:
                if q > p:
                    break
                while t % q == 0:
                    t //= q
            if t == 1:
                k = i
                continue
            # t > 1: check if t is p-smooth
            if t <= p:
                # t is prime (after stripping small primes, if t < 10^6 it's prime)
                # Actually t could be composite with factors > 1000. But if t <= p and we need all factors <= p,
                # and t itself <= p, then even if t is composite, its factors are < t <= p. So it's p-smooth.
                k = i
            else:
                # t > p: need to check if all prime factors of t are <= p
                # t is at most p^2+200, and we've stripped primes <= 1000
                # t could be prime or composite
                if _is_prime_mr(t):
                    # t is prime and > p, so NOT p-smooth
                    break
                else:
                    # Factor t and check
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
    
    # Phase 3: Large primes (5*10^6 to 10^9) - random sampling with fast filter
    if best_k < 200:
        for _ in range(2000):
            if time.time() > deadline:
                break
            # Sample in log space
            log_val = 7 + random.random() * 2  # 10^7 to 10^9
            p = int(10 ** log_val)
            if p % 2 == 0:
                p += 1
            if p > 10**9:
                continue
            # Find next prime
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
                # Strip small primes
                for q in small_primes:
                    if q > p or q * q > t:
                        break
                    while t % q == 0:
                        t //= q
                if t == 1:
                    k = i
                    continue
                if t <= p:
                    k = i
                else:
                    if _is_prime_mr(t):
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
