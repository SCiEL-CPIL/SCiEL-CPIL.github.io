"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import random
from math import gcd
from fractions import Fraction

N = 512
ALL = list(range(1, N))
WEIGHT = {a: Fraction(1, N - a) for a in ALL}
SCORE_ALL = sum(WEIGHT.values())

# Sieve of primes up to N-1
def _sieve(n):
    is_p = [True] * n
    is_p[0] = is_p[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_p[i]:
            for j in range(i * i, n, i):
                is_p[j] = False
    return [i for i in range(2, n) if is_p[i]]

PRIMES = _sieve(N)
PRIME_SET = set(PRIMES)
MAXP = max(PRIMES)
MODS = [a % p for a in ALL for p in PRIMES]

# Precompute, for each prime, the set of a that are NOT divisible by p (coprime to p)
NOTDIV = [set(a for a in ALL if a % p != 0) for p in PRIMES]

def _score(chosen):
    return sum(WEIGHT[a] for a in chosen)

def _greedy(chosen, rng=None):
    chosen = list(chosen)
    chosen_set = set(chosen)
    used = set()
    # Fast coprime filter: a is coprime to every chosen b iff a is coprime to each
    # prime factor of the product of chosen. Track the set of used prime factors.
    for b in chosen:
        for p in PRIMES:
            if b % p == 0:
                used.add(p)
    avail = [a for a in ALL if a not in chosen_set]
    # Order by weight desc (a desc).
    avail.sort(key=lambda a: -a)
    for a in avail:
        # coprime to all chosen?
        ok = True
        for p in used:
            if a % p == 0:
                ok = False
                break
        if ok:
            chosen.append(a)
            chosen_set.add(a)
            for p in PRIMES:
                if a % p == 0:
                    used.add(p)
    return chosen

def _local_search(chosen, rng, deadline, max_iters=20000):
    chosen = list(chosen)
    cur = _score(chosen)
    best = chosen[:]
    best_score = cur
    n = len(chosen)
    # Precompute prime factor set for each a
    for it in range(max_iters):
        if rng is not None and rng.random() < 0.02:
            if it % 400 == 0 and time.time() > deadline:
                break
        # pick a random element to remove
        i = rng.randrange(n) if rng else it % n
        removed = chosen[i]
        new_list = chosen[:i] + chosen[i+1:]
        cand = _greedy(new_list, rng)
        s = _score(cand)
        if s > cur:
            cur = s
            chosen = cand
            n = len(chosen)
            if s > best_score:
                best_score = s
                best = chosen[:]
        # deadline check periodically
        if it % 200 == 0 and time.time() > deadline:
            break
    return best, best_score

import time

def construct():
    start = time.time()
    deadline = start + 110.0  # safety margin before 120s kill
    rng = random.Random(12345)

    # 1) Strong greedy: prefer large a (max weight).
    base = _greedy([])
    best = base
    best_score = _score(best)

    # 2) Local search from base.
    if time.time() < deadline - 5.0:
        cand, cs = _local_search(best, rng, deadline - 5.0, max_iters=30000)
        if cs > best_score:
            best, best_score = cand, cs

    # 3) Randomized restarts: remove a random subset, re-greedy, local search.
    attempt = 0
    while time.time() < deadline - 5.0 and attempt < 40:
        attempt += 1
        # Start from current best, remove k random elements.
        k = rng.randint(1, min(8, len(best)))
        start_list = list(best)
        for _ in range(k):
            if not start_list:
                break
            idx = rng.randrange(len(start_list))
            start_list.pop(idx)
        cand = _greedy(start_list, rng)
        cs = _score(cand)
        if cs > best_score:
            best, best_score = cand, cs
        # quick local search on this candidate
        if time.time() < deadline - 5.0:
            cand2, cs2 = _local_search(cand, rng, deadline - 5.0, max_iters=6000)
            if cs2 > best_score:
                best, best_score = cand2, cs2

    return best
# EVOLVE-BLOCK-END
