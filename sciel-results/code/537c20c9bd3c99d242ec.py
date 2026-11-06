"""Strong program: precompute powerful parts via SPF sieve, scan for max product."""
import time
import math

def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin under 90s

    N = 1000000
    LO = 10000
    HI = N  # n must be in [10000, 1000000], so n+1 <= 1000001

    # Build SPF up to N+1 (we need n+1 up to 1000001)
    limit = N + 1
    spf = list(range(limit + 1))
    if limit >= 0:
        spf[0] = 0
    if limit >= 1:
        spf[1] = 1
    for i in range(2, math.isqrt(limit) + 1):
        if spf[i] == i:
            for j in range(i * i, limit + 1, i):
                if spf[j] == j:
                    spf[j] = i

    # Compute powerful part for each number
    # powerful_part[n] = product of p^e for p^e || n with e >= 2
    pp = [1] * (limit + 1)
    for n in range(2, limit + 1):
        temp = n
        result = 1
        while temp > 1:
            p = spf[temp]
            e = 0
            while temp % p == 0:
                temp //= p
                e += 1
            if e >= 2:
                result *= p ** e
        pp[n] = result

    # Check time
    if time.time() > deadline:
        # Fallback: return a reasonable guess
        return 10000

    # Scan for max pp[n] * pp[n+1]
    best_n = LO
    best_val = pp[LO] * pp[LO + 1]
    for n in range(LO, N):
        v = pp[n] * pp[n + 1]
        if v > best_val:
            best_val = v
            best_n = n
        if time.time() > deadline:
            break

    return best_n
