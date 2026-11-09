"""Search for M with many k such that k*sigma(k) = M, using smooth-integer meet-in-the-middle."""
import time
from collections import defaultdict

def construct():
    start = time.time()
    deadline = start + 140.0  # safety margin before 150s kill

    # Primes up to ~2000 for smooth-number generation
    primes = []
    n = 2000
    is_prime = bytearray(b'\x01') * (n + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            step = i
            start_i = i * i
            is_prime[start_i::step] = bytearray(len(is_prime[start_i::step]))
    for i in range(2, n + 1):
        if is_prime[i]:
            primes.append(i)

    # Sieve sigma up to S for fast sigma lookup
    S = 20000000  # 20 million
    sigma_arr = [0] * (S + 1)
    for i in range(1, S + 1):
        for j in range(i, S + 1, i):
            sigma_arr[j] += i

    def sigma(k):
        if k <= S:
            return sigma_arr[k]
        # Fallback: factorization
        n = k
        s = 1
        p = 2
        while p * p <= n:
            if n % p == 0:
                e = 0
                while n % p == 0:
                    n //= p
                    e += 1
                s *= (p**(e+1) - 1) // (p - 1)
            p += 1 if p == 2 else 2
        if n > 1:
            s *= (n + 1)
        return s

    # Generate smooth numbers using primes up to 47 (first 15 primes)
    smooth_primes = [p for p in primes if p <= 47]
    # smooth_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

    # Generate all 47-smooth numbers up to 10^7
    # Using recursive generation with prime index bounds
    smooth_numbers = []
    max_k = 10_000_000

    def gen(idx, current):
        if current > max_k:
            return
        if idx == len(smooth_primes):
            if current > 1:
                smooth_numbers.append(current)
            return
        p = smooth_primes[idx]
        # Try all powers of p
        val = current
        while val <= max_k:
            gen(idx + 1, val)
            if val > max_k // p:
                break
            val *= p

    gen(0, 1)

    # Compute M = k * sigma(k) for each smooth k, group by M
    m_to_ks = defaultdict(list)
    for k in smooth_numbers:
        if k > max_k:
            continue
        sk = sigma(k)
        M = k * sk
        if M <= 10**18:
            m_to_ks[M].append(k)

    # Also add non-smooth k from the sieve range that might collide
    # For efficiency, only check k up to S where we have sigma
    # But we need to find M values that already have multiple solutions
    # Strategy: for each M with >= 2 solutions from smooth numbers,
    # check if more k in [1, isqrt(M)] also satisfy k*sigma(k) = M

    # First, find candidate M values with many smooth solutions
    candidates = []
    for M, ks in m_to_ks.items():
        if len(ks) >= 2:
            candidates.append((len(ks), M, ks))

    candidates.sort(reverse=True)

    # For top candidates, do a broader search for additional k
    best_M = None
    best_ks = []
    best_count = 0

    # Also do a direct scan: for k in range, compute M and track
    # But 10^7 is too slow for full scan with sigma. Use the smooth set + sieve.
    # Let's do a targeted scan: for each candidate M, check k in 1..isqrt(M)
    # that are NOT in our smooth set but might have k*sigma(k) = M

    import math

    for count, M, ks in candidates[:50]:  # top 50 candidates
        if time.time() > deadline - 10:
            break
        if count <= best_count:
            continue
        limit = math.isqrt(M)
        # ks should all be <= limit; filter
        valid_ks = [k for k in ks if k <= limit and k * sigma(k) == M]
        if len(valid_ks) < 2:
            continue
        # Try to find more k by scanning a subset
        # For efficiency, scan k values where sigma(k) = M/k
        # i.e., k must divide M, and sigma(M//k) ... no, k*sigma(k)=M
        # So sigma(k) = M/k, meaning k | M and sigma(k) = M/k
        # Enumerate divisors of M
        # M can be up to 10^18, factoring might be slow
        # Instead, scan k in ranges around existing solutions
        # Or: for k from 1 to limit, check if k*sigma(k) == M
        # This is O(limit) which is too slow for large limit
        # Alternative: only check k that divide M (since k*sigma(k)=M implies k|M)
        # Enumerate divisors of M up to isqrt(M)
        # Factor M
        n = M
        factors = {}
        p = 2
        temp = n
        while p * p <= temp:
            if temp % p == 0:
                e = 0
                while temp % p == 0:
                    temp //= p
                    e += 1
                factors[p] = e
            p += 1 if p == 2 else 2
        if temp > 1:
            factors[temp] = 1

        # Generate divisors of M up to isqrt(M)
        divisors = [1]
        for p, e in factors.items():
            new_divs = []
            pe = 1
            for i in range(e + 1):
                for d in divisors:
                    val = d * pe
                    if val <= limit:
                        new_divs.append(val)
                pe *= p
            divisors = new_divs

        # Check each divisor k: does k*sigma(k) == M?
        found = set(valid_ks)
        for k in divisors:
            if k in found:
                continue
            if k * sigma(k) == M:
                found.add(k)

        if len(found) > best_count:
            best_count = len(found)
            best_M = M
            best_ks = sorted(found)

    # Fallback: if no good candidate found, use the best from smooth-only
    if best_M is None or best_count < 2:
        if candidates:
            _, M, ks = candidates[0]
            limit = math.isqrt(M)
            valid_ks = [k for k in ks if k <= limit and k * sigma(k) == M]
            if valid_ks:
                best_M = M
                best_ks = sorted(valid_ks)
                best_count = len(valid_ks)
            else:
                best_M = 333312
                best_ks = [336, 372, 434]
                best_count = 3
        else:
            best_M = 333312
            best_ks = [336, 372, 434]
            best_count = 3

    # Final verification
    final_ks = []
    for k in best_ks:
        if isinstance(k, int) and 1 <= k <= math.isqrt(best_M) and k * sigma(k) == best_M:
            final_ks.append(k)
    final_ks = list(set(final_ks))

    if len(final_ks) < 1:
        best_M = 333312
        final_ks = [336, 372, 434]

    return {'M': int(best_M), 'k': [int(x) for x in final_ks]}
