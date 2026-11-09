"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import isqrt

def _gen_powerful(limit):
    max_t = isqrt(limit)
    spf = list(range(max_t + 1))
    for i in range(2, isqrt(max_t) + 1):
        if spf[i] == i:
            for j in range(i * i, max_t + 1, i):
                if spf[j] == j:
                    spf[j] = i
    rad = [1] * (max_t + 1)
    for t in range(2, max_t + 1):
        m = t
        r = 1
        while m > 1:
            q = spf[m]
            r *= q
            while m % q == 0:
                m //= q
        rad[t] = r
    powerful = set()
    for t in range(1, max_t + 1):
        t2 = t * t
        if t2 > limit:
            break
        r = rad[t]
        m = r
        primes = []
        while m > 1:
            q = spf[m]
            primes.append(q)
            while m % q == 0:
                m //= q
        k = len(primes)
        for mask in range(1 << k):
            s = 1
            for i in range(k):
                if (mask >> i) & 1:
                    s *= primes[i]
            val = s * t2
            if val <= limit:
                powerful.add(val)
    return sorted(powerful)

def construct():
    t0 = time.time()
    deadline = t0 + 130.0
    L = 10**9
    P = _gen_powerful(L)
    nP = len(P)
    topn = min(300, nP)
    top = P[nP - topn:]
    candidates = set()
    for i in range(topn):
        for j in range(i, topn):
            s = top[i] + top[j]
            if 2 <= s <= 2 * L:
                candidates.add(s)
    for k in range(0, 300):
        n = L - k * (L // 600)
        if 2 <= n <= 2 * L:
            candidates.add(n)
    candidates = sorted(candidates)
    best_N, best_count, best_pairs = None, 0, []
    for N in candidates:
        if time.time() > deadline - 15:
            break
        i, j = 0, nP - 1
        cnt = 0
        pairs = []
        while i <= j:
            s = P[i] + P[j]
            if s == N:
                if P[i] <= P[j]:
                    pairs.append([P[i], P[j]])
                    cnt += 1
                i += 1
                j -= 1
            elif s < N:
                i += 1
            else:
                j -= 1
        if cnt > best_count:
            best_count = cnt
            best_N = N
            best_pairs = pairs
    if best_N is None:
        best_N, best_pairs = 2, [[1, 1]]
    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
