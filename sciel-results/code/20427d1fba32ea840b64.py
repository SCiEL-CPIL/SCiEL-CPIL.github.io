import math
import time
from collections import defaultdict

def _sigma_slow(n):
    if n == 1:
        return 1
    result = 1
    temp = n
    p = 2
    while p * p <= temp:
        if temp % p == 0:
            exp = 0
            term = 1
            pw = 1
            while temp % p == 0:
                temp //= p
                exp += 1
                pw *= p
                term += pw
            result *= term
        p += 1 if p == 2 else 2
    if temp > 1:
        result *= (1 + temp)
    return result

def construct():
    start = time.time()
    deadline = start + 135.0  # safety margin below 150s wall-clock

    # Fallback: a known small example with 3 solutions.
    # M = 333312, k in {336, 372, 434}
    def _fallback():
        M = 333312
        ks = [336, 372, 434]
        # verify quickly
        ok = []
        for k in ks:
            if k * _sigma_slow(k) == M:
                ok.append(k)
        if len(ok) >= 1:
            return {'M': int(M), 'k': [int(x) for x in ok]}
        # absolute minimal: k=1 -> M=1, sigma(1)=1, 1*1=1
        return {'M': 1, 'k': [1]}

    # Try a fast sigma sieve.
    N = 10_000_000
    # sigma array
    sigma = [0] * (N + 1)
    for d in range(1, N + 1):
        for j in range(d, N + 1, d):
            sigma[j] += d

    if time.time() > deadline - 8.0:
        return _fallback()

    # Build map from M -> list of k
    M_to_ks = defaultdict(list)
    for k in range(1, N + 1):
        M_val = k * sigma[k]
        if M_val <= 10**18:
            M_to_ks[M_val].append(k)
        if time.time() > deadline - 8.0:
            break

    # Pick the M with the most k's (with k <= sqrt(M))
    best_M = None
    best_sols = None
    for M_val, ks in M_to_ks.items():
        if len(ks) < 2:
            continue
        limit = math.isqrt(M_val)
        valid = [k for k in ks if k <= limit]
        if best_sols is None or len(valid) > len(best_sols):
            best_sols = valid
            best_M = M_val
        if time.time() > deadline - 6.0:
            break

    if best_sols is None or len(best_sols) < 2:
        return _fallback()

    # Final exact verification with the sieve values (k <= N guaranteed)
    verified = []
    for k in best_sols:
        if 1 <= k <= math.isqrt(best_M) and k <= N:
            if k * sigma[k] == best_M:
                verified.append(k)

    if not verified:
        return _fallback()

    verified = sorted(set(verified))
    return {'M': int(best_M), 'k': [int(x) for x in verified]}
