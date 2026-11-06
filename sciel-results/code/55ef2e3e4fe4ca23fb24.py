import math
import time
import random

def construct():
    start_time = time.time()
    DEADLINE = start_time + 115.0  # safety margin before 150s limit

    # Precompute small primes up to 1000 for quick trial division
    small_primes = []
    _is_p = [True] * 1001
    _is_p[0] = _is_p[1] = False
    for _i in range(2, 32):
        if _is_p[_i]:
            for _j in range(_i * _i, 1001, _i):
                _is_p[_j] = False
    small_primes = [i for i in range(2, 1001) if _is_p[i]]

    # Deterministic Miller-Rabin for n < 3,474,749,660,383
    def is_prime(n):
        if n < 2:
            return False
        if n < 4:
            return True
        if n % 2 == 0 or n % 3 == 0:
            return False
        for a in [2, 3, 5, 7, 11, 13, 17]:
            if a >= n:
                continue
            d = n - 1
            r = 0
            while d % 2 == 0:
                d //= 2
                r += 1
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

    # Pollard's Rho algorithm for finding a factor
    def pollard_rho(n):
        if n % 2 == 0:
            return 2
        if n % 3 == 0:
            return 3
        while True:
            x = random.randint(2, n - 1)
            y = x
            c = random.randint(1, n - 1)
            d = 1
            while d == 1:
                x = (x * x + c) % n
                y = (y * y + c) % n
                y = (y * y + c) % n
                d = math.gcd(abs(x - y), n)
            if d != n:
                return d

    # Check if n has all prime factors <= p
    # Returns True if n is p-smooth, False otherwise
    def is_p_smooth(n, p):
        if n <= 1:
            return True
        # Strip small primes up to min(p, 1000)
        rem = n
        for sp in small_primes:
            if sp > p:
                break
            while rem % sp == 0:
                rem //= sp
        if rem == 1:
            return True
        if rem <= p:
            return True
        # Need to check if rem has any prime factor > p
        # Use Pollard-Rho to factor rem
        factors = []
        stack = [rem]
        while stack:
            m = stack.pop()
            if m == 1:
                continue
            if is_prime(m):
                factors.append(m)
            else:
                d = pollard_rho(m)
                stack.append(d)
                stack.append(m // d)
        for f in factors:
            if f > p:
                return False
        return True

    # Check run length for prime p
    def check_run(p, k_max=200):
        k = 0
        p_sq = p * p
        for i in range(1, k_max + 1):
            if time.time() > DEADLINE - 5:
                break
            n = p_sq + i
            if is_p_smooth(n, p):
                k += 1
            else:
                break
        return k

    # Segmented sieve to get primes in [lo, hi]
    def segmented_sieve(lo, hi):
        if hi < 2:
            return []
        if lo < 2:
            lo = 2
        size = hi - lo + 1
        is_p = [True] * size
        limit = int(math.isqrt(hi))
        # Sieve small primes up to sqrt(hi)
        small_lim = limit + 1
        is_small = [True] * small_lim
        small_primes_list = []
        for i in range(2, small_lim):
            if is_small[i]:
                small_primes_list.append(i)
                for j in range(i * i, small_lim, i):
                    is_small[j] = False
        for sp in small_primes_list:
            # Mark multiples of sp in [lo, hi]
            start = ((lo + sp - 1) // sp) * sp
            if start == sp:
                start += sp
            for j in range(start, hi + 1, sp):
                is_p[j - lo] = False
        return [lo + i for i in range(size) if is_p[i]]

    best_k = 0
    best_p = 2

    # Phase 1: Search small primes with full k range
    small_primes_list = segmented_sieve(2, 50000)
    for p in small_primes_list:
        if time.time() > DEADLINE - 15:
            break
        k = check_run(p)
        if k > best_k:
            best_k = k
            best_p = p

    # Phase 2: Search larger primes with reduced k range
    # For larger p, p^2 is huge, factorization is expensive
    # Use segmented sieve in chunks
    large_ranges = [
        (50000, 200000),
        (200000, 1000000),
        (1000000, 5000000),
        (5000000, 20000000),
        (20000000, 100000000),
        (100000000, 500000000),
        (500000000, 1000000000),
    ]

    for lo, hi in large_ranges:
        if time.time() > DEADLINE - 20:
            break
        # Limit the number of candidates we test in this range
        # For very large ranges, sample a subset
        candidates = segmented_sieve(lo, hi)
        # If too many candidates, sample
        if len(candidates) > 5000:
            # Take evenly spaced samples
            step = len(candidates) // 5000 + 1
            candidates = candidates[::step]

        for p in candidates:
            if time.time() > DEADLINE - 10:
                break
            # For larger p, limit k check based on remaining time
            remaining = DEADLINE - time.time()
            # Estimate time per check: for large p, each check might take ~0.01-0.1s
            # Be conservative
            max_k_check = min(200, max(1, int(remaining / 0.05)))
            if max_k_check < 1:
                break
            k = check_run(p, k_max=max_k_check)
            if k > best_k:
                best_k = k
                best_p = p

    # Ensure valid result
    if best_k < 1:
        best_k = 1
        best_p = 2

    return {'p': best_p, 'k': best_k}
