# EVOLVE-BLOCK-START
import time
from math import isqrt
import bisect
from collections import Counter

_DEADLINE = 150.0
_START = time.time()

def _generate_powerful(limit):
    """Generate all powerful numbers <= limit via DFS over primes."""
    if limit < 1:
        return [1]
    sqrt_lim = isqrt(limit)
    if sqrt_lim < 2:
        return [1]
    sieve = bytearray(b'\x01') * (sqrt_lim + 1)
    sieve[0] = sieve[1] = 0
    for i in range(2, isqrt(sqrt_lim) + 1):
        if sieve[i]:
            sieve[i*i::i] = bytearray(len(sieve[i*i::i]))
    primes = [i for i in range(2, sqrt_lim + 1) if sieve[i]]
    result = []
    def dfs(start_idx, current):
        if time.time() - _START > _DEADLINE - 5:
            return
        result.append(current)
        for i in range(start_idx, len(primes)):
            p = primes[i]
            p2 = p * p
            val = current * p2
            if val > limit:
                break
            dfs(i + 1, val)
            val *= p
            while val <= limit:
                dfs(i + 1, val)
                val *= p
    dfs(0, 1)
    return result

def _count_pairs(N, powerful, powerful_set):
    half = N // 2
    idx = bisect.bisect_right(powerful, half)
    cnt = 0
    pairs = []
    for i in range(idx):
        a = powerful[i]
        b = N - a
        if b in powerful_set:
            cnt += 1
            pairs.append([a, b])
    return cnt, pairs

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE - 12
    limit_p = 10_000_000_000
    powerful = _generate_powerful(limit_p)
    powerful.sort()
    powerful_set = set(powerful)
    n_pow = len(powerful)
    if time.time() > deadline:
        return {'N': 2, 'pairs': [[1, 1]]}
    best_N = 2
    best_count = 0
    best_pairs = [[1, 1]]
    # Phase 1: candidate generation from small powerful numbers
    K = min(3000, n_pow)
    small_pow = powerful[:K]
    sum_counter = Counter()
    for i in range(K):
        a = small_pow[i]
        for j in range(i, K):
            sum_counter[a + small_pow[j]] += 1
    top_candidates = sum_counter.most_common(500)
    # Phase 2: scan small N
    for N in range(2, 50_001):
        if time.time() > deadline:
            break
        cnt, pairs = _count_pairs(N, powerful, powerful_set)
        if cnt > best_count:
            best_count = cnt
            best_N = N
            best_pairs = pairs
    # Phase 3: verify top candidates
    for N, freq in top_candidates:
        if time.time() > deadline:
            break
        if N < 2 or N > 2 * limit_p:
            continue
        cnt, pairs = _count_pairs(N, powerful, powerful_set)
        if cnt > best_count:
            best_count = cnt
            best_N = N
            best_pairs = pairs
    # Phase 4: scan around best and larger N
    N_lo = max(2, best_N - 200_000)
    N_hi = min(2 * limit_p, best_N + 200_000)
    for N in range(N_lo, N_hi + 1):
        if time.time() > deadline:
            break
        cnt, pairs = _count_pairs(N, powerful, powerful_set)
        if cnt > best_count:
            best_count = cnt
            best_N = N
            best_pairs = pairs
    # Phase 5: scan larger N
    N_start = max(N_hi + 1, 100_000)
    N_end = min(2 * limit_p, 1_000_000_000)
    for N in range(N_start, N_end + 1):
        if time.time() > deadline:
            break
        cnt, pairs = _count_pairs(N, powerful, powerful_set)
        if cnt > best_count:
            best_count = cnt
            best_N = N
            best_pairs = pairs
    if not best_pairs:
        best_pairs = [[1, 1]]
        best_N = 2
    return {'N': best_N, 'pairs': best_pairs}
# EVOLVE-BLOCK-END
