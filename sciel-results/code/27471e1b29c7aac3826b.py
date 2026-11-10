import time
import math

def _is_prime(n):
    if n < 2:
        return False
    if n < 9:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def _primes_in_range(lo, hi):
    out = []
    x = lo
    if x % 2 == 0:
        x += 1
    for cand in range(x, hi + 1, 2):
        if _is_prime(cand):
            out.append(cand)
    return out

def _tau_perp(primes):
    k = len(primes)
    if k == 0:
        return 0
    # Build all 2^k divisors with their prime-mask, then sort by value.
    # Divisor value = product of primes in mask.
    # We sort by value (numeric), which for large primes correlates with mask bit pattern.
    divs = []  # list of (value, mask)
    for m in range(1 << k):
        v = 1
        mm = m
        idx = 0
        while mm:
            if mm & 1:
                v *= primes[idx]
            mm >>= 1
            idx += 1
        divs.append((v, m))
    divs.sort(key=lambda t: t[0])
    # Count adjacent pairs with gcd(value_i, value_{i+1}) == 1
    # For squarefree, gcd==1 iff (mask_i & mask_{i+1}) == 0
    score = 0
    for i in range(len(divs) - 1):
        if (divs[i][1] & divs[i + 1][1]) == 0:
            score += 1
    return score

def construct():
    t0 = time.time()
    DEADLINE = 120.0

    # Baseline: 15th primorial (first 15 primes)
    first15 = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    best_n = 1
    for p in first15:
        best_n *= p
    best_score = _tau_perp(first15)

    # Candidate: 15 consecutive primes starting around 101
    # 101..149 are 15 primes (101,103,107,109,113,127,131,137,139,149,...)
    # Let's find 15 consecutive primes in [101, 170]
    primes_range = _primes_in_range(101, 170)
    # Take first 15
    cand15 = primes_range[:15]
    prod = 1
    for p in cand15:
        prod *= p
    if prod <= 10**18 and prod >= 2:
        s = _tau_perp(cand15)
        if s > best_score:
            best_score = s
            best_n = prod

    # Also try 15 primes starting at various offsets in the range
    for start in range(len(primes_range) - 14):
        cand = primes_range[start:start + 15]
        prod = 1
        for p in cand:
            prod *= p
        if prod > 10**18 or prod < 2:
            continue
        if time.time() - t0 > DEADLINE - 15:
            break
        s = _tau_perp(cand)
        if s > best_score:
            best_score = s
            best_n = prod

    return best_n
