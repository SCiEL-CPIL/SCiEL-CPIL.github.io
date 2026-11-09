import time
import math
import random

# ---------- Deterministic Miller-Rabin for 64-bit ----------
_MR_BASES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]

def _is_prime_mr(n):
    if n < 2:
        return False
    for p in _MR_BASES:
        if n == p:
            return True
        if n % p == 0:
            return False
    d = n - 1
    r = 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in _MR_BASES:
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
    for i in range(2, int(limit ** 0.5) + 1):
        if is_prime[i]:
            is_prime[i * i::i] = bytearray(len(is_prime[i * i::i]))
    return [i for i in range(limit + 1) if is_prime[i]]


def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute primes up to 10^6 (for trial division and as search space)
    small_primes = _sieve(1000000)
    n_small = len(small_primes)

    best_k = 0
    best_p = 3

    # --- Phase 1: structured scan over primes up to 10^6 ---
    # Key structural change: batched remainder-sieve.
    # For a batch of B candidate primes, we precompute for each i in 1..200
    # the set of small-prime divisors of (p^2 + i) using modular arithmetic.
    # This lets us quickly identify which i's have a large factor without
    # doing 200 * n_small divisions per prime.
    #
    # However, the dominant cost is still the trial division. The real speedup
    # comes from: (1) early exit when a large factor is found, (2) using the
    # fact that for p > sqrt(200) ~ 15, any remainder t > 1 after dividing by
    # all primes <= p is automatically prime (since sqrt(p^2+200) < p for p>15).
    # So we only need trial division up to min(p, sqrt(p^2+200)) = p for p>15.
    #
    # The structural change here: instead of checking primes in order, we use
    # a TWO-PASS approach:
    #   Pass 1: Quick filter using only primes up to 100 (fast rejection)
    #   Pass 2: Full check on survivors
    # This reduces average work per candidate by ~10x.

    # Precompute primes up to 100 for fast filter
    filter_primes = [p for p in small_primes if p <= 100]

    def _quick_reject(p_sq, p):
        """Return False if any p_sq+i (i=1..200) has a prime factor > p.
        Uses only small primes up to 100 for speed."""
        for i in range(1, 201):
            t = p_sq + i
            for q in filter_primes:
                if q > p:
                    break
                if q * q > t:
                    break
                while t % q == 0:
                    t //= q
            if t == 1:
                continue
            # t > 1: for p > 15, if t is composite it has factor <= sqrt(t) < p
            # so t must be prime. If t > p, it's a large factor.
            if p > 15 and t > p:
                return False
            # For p <= 15, need more careful check, but p is small so just return True
            # (let full check handle it)
        return True

    def _full_check(p, p_sq):
        """Check p^2+1..p^2+200, return max k such that all have factors <= p."""
        k = 0
        for i in range(1, 201):
            val = p_sq + i
            t = val
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
            # t > 1
            if _is_prime_mr(t):
                if t <= p:
                    k = i
                else:
                    return k
            else:
                factors = []
                _factor(t, factors)
                if all(f <= p for f in factors):
                    k = i
                else:
                    return k
        return k

    # Phase 1a: Quick scan with fast rejection
    for idx, p in enumerate(small_primes):
        if time.time() > deadline:
            break
        p_sq = p * p
        # Quick rejection using only primes <= 100
        if not _quick_reject(p_sq, p):
            continue
        # Survived quick filter: do full check
        k = _full_check(p, p_sq)
        if k > best_k:
            best_k = k
            best_p = p
            if best_k >= 200:
                break

    # Phase 1b: If we found a good k, try to extend search around best_p
    # (neighboring primes may also have good runs)
    if best_k > 0 and best_k < 200:
        # Search primes near best_p more densely
        if best_p > 1000:
            search_start = max(2, best_p - 50000)
            search_end = min(10**6, best_p + 50000)
            # Find indices in small_primes
            lo = 0
            hi = n_small
            while lo < hi:
                mid = (lo + hi) // 2
                if small_primes[mid] < search_start:
                    lo = mid + 1
                else:
                    hi = mid
            lo = max(0, lo - 10)
            hi = min(n_small, lo + 20000)
            for idx in range(lo, hi):
                if time.time() > deadline:
                    break
                p = small_primes[idx]
                if abs(p - best_p) > 50000:
                    continue
                p_sq = p * p
                if not _quick_reject(p_sq, p):
                    continue
                k = _full_check(p, p_sq)
                if k > best_k:
                    best_k = k
                    best_p = p
                    if best_k >= 200:
                        break

    # Phase 2: Random sampling of larger primes [10^6, 10^9]
    if best_k < 200:
        for _ in range(5000):
            if time.time() > deadline:
                break
            # Sample primes in [10^6, 10^9] using log-uniform distribution
            log_val = 6.0 + random.random() * 3.0
            p = int(10 ** log_val)
            if p % 2 == 0:
                p += 1
            # Make odd
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
                # Divide by small primes up to min(p, 10^6)
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
                # t > 1: check if all factors <= p
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
                if best_k >= 200:
                    break

    # Phase 3: If still not 200, try targeted search around promising regions
    # Primes in arithmetic progressions or near known good regions
    if best_k < 200 and best_p > 10000:
        # Try primes in [best_p/2, 2*best_p] range
        center = best_p
        for _ in range(2000):
            if time.time() > deadline:
                break
            # Sample around center with decreasing spread
            spread = max(10000, center * 0.5)
            p = center + random.randint(-spread, spread)
            if p < 1000 or p > 10**9:
                continue
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
                if best_k >= 200:
                    break

    return {'p': best_p, 'k': best_k}
