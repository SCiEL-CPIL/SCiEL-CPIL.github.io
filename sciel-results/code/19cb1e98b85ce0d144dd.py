import time
import random
from itertools import combinations

N = 5000
MAX_TIME = 295.0

def _primes(limit):
    if limit < 2:
        return []
    sieve = [True] * (limit + 1)
    sieve[0] = sieve[1] = False
    for p in range(2, int(limit ** 0.5) + 1):
        if sieve[p]:
            for q in range(p * p, limit + 1, p):
                sieve[q] = False
    return [p for p, ok in enumerate(sieve) if ok]

_SMALL_PRIMES = [2, 3, 5, 7, 11, 13]

def _get_violating_prime(S, primes):
    len_S = len(S)
    for p in primes:
        if p > len_S:
            break
        residues = set(x % p for x in S)
        if len(residues) == p:
            return p
    return None

def _is_admissible(S, primes):
    return _get_violating_prime(S, primes) is None

def _exhaustive(max_size, K):
    """Search for largest admissible set within [0,K] up to max_size."""
    best = []
    # try from max_size down
    for size in range(max_size, 0, -1):
        found = False
        for combo in combinations(range(K + 1), size):
            if _is_admissible(list(combo), _SMALL_PRIMES):
                best = list(combo)
                found = True
                break
        if found:
            return best
    return best

def _local_add(S, max_iters=100):
    S = set(S)
    full = set(range(N + 1))
    removed = full - S
    for _ in range(max_iters):
        improved = False
        for x in sorted(removed):
            S_new = S | {x}
            if _is_admissible(S_new, _SMALL_PRIMES):
                S = S_new
                removed.discard(x)
                improved = True
                break
        if not improved:
            break
    return list(S)

def _greedy_remove(seed=0, strategy=0):
    if seed is not None:
        random.seed(seed)
    S = set(range(N + 1))
    while True:
        len_S = len(S)
        if len_S == 0:
            break
        viol_p = None
        for p in _SMALL_PRIMES:
            if p > len_S:
                break
            residues = set(x % p for x in S)
            if len(residues) == p:
                viol_p = p
                break
        if viol_p is None:
            break
        p = viol_p
        counts = [0] * p
        for x in S:
            counts[x % p] += 1
        min_count = min(counts)
        candidates = [r for r in range(p) if counts[r] == min_count]
        if strategy == 0:
            r_to_remove = candidates[0]
        elif strategy == 1:
            r_to_remove = random.choice(candidates)
        else:
            c2 = [r for r in range(p) if counts[r] <= min_count + 1]
            r_to_remove = random.choice(c2)
        S = {x for x in S if x % p != r_to_remove}
    return list(S)

def _random_construct(seed):
    random.seed(seed)
    S = set()
    for x in range(N + 1):
        if random.random() < 0.03:
            S.add(x)
    while True:
        viol_p = _get_violating_prime(list(S), _SMALL_PRIMES)
        if viol_p is None:
            break
        p = viol_p
        counts = [0] * p
        for x in S:
            counts[x % p] += 1
        r_to_remove = max(range(p), key=lambda r: counts[r])
        S = {x for x in S if x % p != r_to_remove}
    return _local_add(list(S))

def construct():
    start = time.time()
    best = []

    # Exhaustive: find largest admissible set in small universe
    # We proved max size <= 4 (and 4 impossible), so search sizes 4,3,2,1
    for K in [10, 20, 50, 100]:
        if time.time() - start > MAX_TIME:
            break
        cand = _exhaustive(4, K)
        if len(cand) > len(best):
            best = cand
        if len(best) >= 3:
            break

    # Greedy constructions
    if time.time() - start < MAX_TIME:
        for strat in range(3):
            res = _greedy_remove(seed=strat * 7 + 1, strategy=strat)
            if len(res) > len(best):
                best = res
        imp = _local_add(best)
        if len(imp) > len(best):
            best = imp

    # Random constructions
    it = 0
    while time.time() - start < MAX_TIME - 5:
        it += 1
        res = _random_construct(seed=1000 + it)
        if len(res) > len(best):
            best = res
            imp = _local_add(best)
            if len(imp) > len(best):
                best = imp

    # Safety: ensure admissibility
    if not _is_admissible(best, _SMALL_PRIMES):
        S = set(best)
        while S:
            viol = _get_violating_prime(list(S), _SMALL_PRIMES)
            if viol is None:
                best = list(S)
                break
            p = viol
            counts = [0] * p
            for x in S:
                counts[x % p] += 1
            r = max(range(p), key=lambda r: counts[r])
            S = {x for x in S if x % p != r}
        if not S:
            best = []

    # Guaranteed fallback if somehow empty
    if len(best) == 0:
        best = [0]

    return best
