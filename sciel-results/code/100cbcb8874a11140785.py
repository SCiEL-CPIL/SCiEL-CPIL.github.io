import time
import math
import random

# Precompute smallest prime factor up to 20000
_MAX_SUM = 20000
_spf = list(range(_MAX_SUM + 1))
for _i in range(2, int(_MAX_SUM ** 0.5) + 1):
    if _spf[_i] == _i:
        for _j in range(_i * _i, _MAX_SUM + 1, _i):
            if _spf[_j] == _j:
                _spf[_j] = _i

def _factor(n):
    """Return set of distinct prime factors of n using SPF."""
    s = set()
    while n > 1:
        p = _spf[n]
        s.add(p)
        while n % p == 0:
            n //= p
    return s

# Precompute prime factor sets for all possible sums 2..20000
_pf_cache = [None] * (_MAX_SUM + 1)
for _x in range(2, _MAX_SUM + 1):
    _pf_cache[_x] = _factor(_x)

def _score_full(arr):
    """Full score: number of distinct primes across all pairwise sums."""
    primes = set()
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            primes |= _pf_cache[arr[i] + arr[j]]
    return len(primes)

def _local_search(arr, deadline, rng):
    """Local search via single-element swaps. Returns best (score, arr)."""
    n = len(arr)
    arr = list(arr)
    cur_score = _score_full(arr)
    best_score = cur_score
    best_arr = list(arr)

    # Precompute for each index i, the set of primes from sums involving i
    # and a global counter of how many sums each prime appears in
    def build_state(a):
        # prime_counts[p] = number of sums that have prime p
        prime_counts = {}
        # per-index prime sets (sums involving index i)
        idx_primes = [set() for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                s = a[i] + a[j]
                for p in _pf_cache[s]:
                    prime_counts[p] = prime_counts.get(p, 0) + 1
                    idx_primes[i].add(p)
                    idx_primes[j].add(p)
        # score = number of primes with count > 0
        sc = sum(1 for c in prime_counts.values() if c > 0)
        return prime_counts, idx_primes, sc

    prime_counts, idx_primes, cur_score = build_state(arr)
    best_score = cur_score
    best_arr = list(arr)

    # Candidate pool: multiples of 6, 12, powers of 2, small numbers
    pool = set()
    for m in range(1, 1667):  # up to 10000
        pool.add(m * 6)
    for m in range(1, 834):
        pool.add(m * 12)
    p = 1
    while p <= 10000:
        pool.add(p)
        p *= 2
    # Also add some structured sets
    for base in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for k in range(1, 1001):
            v = base * k
            if v <= 10000:
                pool.add(v)
    pool = sorted(pool)
    pool_set = set(pool)

    # Remove current elements from pool for swap candidates
    cur_set = set(arr)

    no_improve = 0
    max_no_improve = 20000

    while time.time() < deadline and no_improve < max_no_improve:
        # Pick a random index to swap
        i = rng.randint(0, n - 1)
        old_val = arr[i]

        # Find a candidate not in cur_set
        # Try a few random candidates
        found = False
        for _attempt in range(50):
            cand = pool[rng.randint(0, len(pool) - 1)]
            if cand != old_val and cand not in cur_set:
                found = True
                break
        if not found:
            no_improve += 1
            continue

        # Swap: remove old_val at index i, add cand
        # Update prime_counts and idx_primes
        # Sums involving i: (i, j) for all j != i
        # Old sums: old_val + arr[j]
        # New sums: cand + arr[j]

        # Decrement old primes
        for j in range(n):
            if j == i:
                continue
            s_old = old_val + arr[j]
            s_new = cand + arr[j]
            for p in _pf_cache[s_old]:
                prime_counts[p] -= 1
                if prime_counts[p] == 0:
                    cur_score -= 1
                idx_primes[i].discard(p)
                idx_primes[j].discard(p)
            for p in _pf_cache[s_new]:
                if prime_counts.get(p, 0) == 0:
                    cur_score += 1
                prime_counts[p] = prime_counts.get(p, 0) + 1
                idx_primes[i].add(p)
                idx_primes[j].add(p)

        # Accept if improved or equal
        if cur_score <= best_score:
            if cur_score < best_score:
                best_score = cur_score
                best_arr = list(arr)
                no_improve = 0
            else:
                no_improve += 1
            # Commit the swap
            arr[i] = cand
            cur_set.discard(old_val)
            cur_set.add(cand)
        else:
            # Revert
            for j in range(n):
                if j == i:
                    continue
                s_old = old_val + arr[j]
                s_new = cand + arr[j]
                for p in _pf_cache[s_new]:
                    prime_counts[p] -= 1
                    if prime_counts[p] == 0:
                        cur_score -= 1
                    idx_primes[i].discard(p)
                    idx_primes[j].discard(p)
                for p in _pf_cache[s_old]:
                    if prime_counts.get(p, 0) == 0:
                        cur_score += 1
                    prime_counts[p] = prime_counts.get(p, 0) + 1
                    idx_primes[i].add(p)
                    idx_primes[j].add(p)
            no_improve += 1

    return best_score, best_arr

def construct():
    start = time.time()
    deadline = start + 85.0
    rng = random.Random(42)

    n = 12
    best_score = 10**9
    best_arr = None

    # Generate diverse seeds
    seeds = []

    # Seed 1: multiples of 6
    s1 = list(range(6, 6 * n + 1, 6))
    seeds.append(s1)

    # Seed 2: multiples of 12
    s2 = list(range(12, 12 * n + 1, 12))
    seeds.append(s2)

    # Seed 3: powers of 2
    s3 = []
    p = 1
    while len(s3) < n:
        s3.append(p)
        p *= 2
    seeds.append(s3)

    # Seed 4: small consecutive
    s4 = list(range(1, n + 1))
    seeds.append(s4)

    # Seed 5: multiples of 3
    s5 = list(range(3, 3 * n + 1, 3))
    seeds.append(s5)

    # Seed 6: random from pool
    pool = set()
    for m in range(1, 1667):
        pool.add(m * 6)
    for m in range(1, 834):
        pool.add(m * 12)
    p = 1
    while p <= 10000:
        pool.add(p)
        p *= 2
    pool = sorted(pool)
    for _ in range(5):
        s = rng.sample(pool, n)
        seeds.append(sorted(s))

    # Evaluate and local-search each seed
    for seed in seeds:
        if time.time() > deadline - 5.0:
            break
        # Ensure distinct and <= 10000
        if len(set(seed)) != n or any(x < 1 or x > 10000 for x in seed):
            continue
        sc, arr = _local_search(seed, deadline, rng)
        if sc < best_score:
            best_score = sc
            best_arr = list(arr)

    if best_arr is None:
        return list(range(1, 13))
    return best_arr
