"""Powerful-number pair representations, frequency-based candidate search."""
import time
import random
from math import isqrt

_START = time.time()
_DEADLINE = _START + 135.0  # safety margin before 150s kill

def _time_left():
    return _DEADLINE - time.time()

def _is_powerful(n):
    """Exact check: every prime factor exponent >= 2 (or n==1)."""
    if n == 1:
        return True
    x = n
    d = 2
    while d * d <= x:
        if x % d == 0:
            e = 0
            while x % d == 0:
                x //= d
                e += 1
            if e == 1:
                return False
        d += 1 if d == 2 else 2
    if x > 1:
        return False  # leftover prime with exponent 1
    return True

def _sieve_primes(n):
    """Return list of primes <= n using a bytearray sieve."""
    if n < 2:
        return []
    size = n + 1
    sieve = bytearray([1]) * size
    sieve[0:2] = b'\x00\x00'
    for i in range(2, isqrt(n) + 1):
        if sieve[i]:
            sieve[i * i::i] = bytearray(len(sieve[i * i::i]))
    return [i for i in range(2, size) if sieve[i]]

def _gen_powerful(limit, primes):
    """Generate all powerful numbers <= limit via recursive prime-power construction."""
    result = [1]
    n = len(primes)
    def rec(idx, cur):
        if idx >= n or cur > limit:
            return
        p = primes[idx]
        v = cur
        # multiply by p^e for e>=2 (powerful requires exponent >= 2 for each prime used)
        while v <= limit:
            result.append(v)
            nxt = v * p * p
            if nxt > limit:
                break
            v = nxt
        rec(idx + 1, cur)
    rec(0, 1)
    return sorted(set(result))

def _count_pairs_for_n(N, pw):
    """Two-pointer count of pairs a<=b, a+b=N, both in pw. Returns (count, pairs)."""
    i = 0
    j = len(pw) - 1
    count = 0
    pairs = []
    while i <= j:
        s = pw[i] + pw[j]
        if s == N:
            if pw[i] <= pw[j]:
                count += 1
                pairs.append([pw[i], pw[j]])
            i += 1
            j -= 1
        elif s < N:
            i += 1
        else:
            j -= 1
    return count, pairs

def construct():
    # --- Phase 1: generate powerful numbers (once, deterministic) ---
    if _time_left() < 30:
        # fallback: small brute force
        limit = 20000
        powerful = sorted({a * a * b ** 3 for b in range(1, 28)
                           for a in range(1, isqrt(limit // b ** 3) + 1)})
        best = max(range(2, limit + 1),
                   key=lambda n: sum(1 for a in powerful if 2 * a <= n and n - a in set(powerful)))
        return {'N': best,
                'pairs': [[a, best - a] for a in powerful
                          if 2 * a <= best and best - a in set(powerful)]}

    MAXN = 10**18
    primes = _sieve_primes(10**9)  # ~50M primes, sieve ~few seconds
    if _time_left() < 60:
        # reduce prime bound if running low
        primes = _sieve_primes(10**7)

    pw = _gen_powerful(MAXN, primes)
    pw_set = set(pw)
    L = len(pw)
    print(f"Generated {L} powerful numbers", flush=True)

    if _time_left() < 15:
        # not enough time for sampling; fall back to a known small search
        small = [x for x in pw if x <= 10**7]
        best, best_pairs = 2, [[1, 1]]
        best_c = 1
        for N in range(2, 10**7 + 1):
            if _time_left() < 5:
                break
            c = sum(1 for a in small if 2 * a <= N and (N - a) in pw_set)
            if c > best_c:
                best_c = c
                best = N
                best_pairs = [[a, N - a] for a in small if 2 * a <= N and (N - a) in pw_set]
        return {'N': best, 'pairs': best_pairs}

    # --- Phase 2: frequency-based candidate sampling ---
    random.seed(12345)
    freq = {}
    sample_budget = int(min(2_000_000, max(200_000, _time_left() * 20000)))
    for _ in range(sample_budget):
        if _time_left() < 20:
            break
        i = random.randrange(L)
        j = random.randrange(L)
        if i > j:
            i, j = j, i
        s = pw[i] + pw[j]
        if s <= MAXN:
            freq[s] = freq.get(s, 0) + 1

    # top candidates by frequency
    top = sorted(freq.items(), key=lambda kv: -kv[1])
    candidates = [s for s, c in top[:400] if s >= 2]
    # also include some structured candidates (small N with many reps)
    small_candidates = [n for n in range(2, 10**6) if n not in freq]
    candidates += small_candidates[:200]

    # --- Phase 3: evaluate candidates, verify pairs ---
    best_N = 2
    best_pairs = [[1, 1]]
    best_count = 1

    for N in candidates:
        if _time_left() < 10:
            break
        if N < 2 or N > MAXN:
            continue
        count, pairs = _count_pairs_for_n(N, pw)
        if count > best_count:
            # verify each pair by exact factorization
            verified = []
            for a, b in pairs:
                if _is_powerful(a) and _is_powerful(b) and a + b == N and a <= b:
                    verified.append([a, b])
            if len(verified) > best_count:
                best_count = len(verified)
                best_N = N
                best_pairs = verified

    return {'N': best_N, 'pairs': best_pairs}
