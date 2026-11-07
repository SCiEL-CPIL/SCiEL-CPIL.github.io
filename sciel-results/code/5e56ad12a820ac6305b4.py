"""Rewritten program: search for max set in 1..128 with no triple whose
three pair-sums are all in the set. Returns a list of Python ints."""
import time
import itertools
import random

_N = 128
_DEADLINE_FRAC = 0.85  # safety margin vs 90s wall-clock kill

def _is_valid(S):
    Sset = set(S)
    for a, b, c in itertools.combinations(S, 3):
        if a + b in Sset and a + c in Sset and b + c in Sset:
            return False
    return True

def _count_bad(S):
    Sset = set(S)
    cnt = 0
    for a, b, c in itertools.combinations(S, 3):
        if a + b in Sset and a + c in Sset and b + c in Sset:
            cnt += 1
    return cnt

def _greedy_grow(S, order):
    S = set(S)
    for x in order:
        if x not in S and _is_valid(S | {x}):
            S.add(x)
    return S

def _local_search(S, iters, rng, deadline):
    S = set(S)
    n = len(S)
    for _ in range(iters):
        if time.time() > deadline:
            break
        x = rng.randint(1, _N)
        if x not in S and _is_valid(S | {x}):
            S.add(x)
            n = len(S)
    return S

def _seed_set():
    # upper half: always valid, size 64
    return set(range(_N // 2 + 1, _N + 1))

def construct():
    t0 = time.time()
    deadline = t0 + 90.0 * _DEADLINE_FRAC

    best = _seed_set()
    best_n = len(best)

    rng = random.Random(12345)
    # Try multiple randomized starts: upper half with some removals,
    # then greedy grow + local search.
    trial = 0
    while time.time() < deadline:
        trial += 1
        start = _seed_set()
        # remove a few random elements to allow small-number insertion
        k = rng.randint(0, 8)
        for _ in range(k):
            start.discard(rng.choice(sorted(start)))
        S = _greedy_grow(start, list(range(1, _N + 1)))
        S = _local_search(S, 4000, rng, deadline)
        if _is_valid(S) and len(S) > best_n:
            best = S
            best_n = len(S)
            if best_n >= 70:  # strong result, stop early
                break

    # Final safety: ensure validity; fall back to seed if broken.
    if not _is_valid(best):
        best = _seed_set()

    return [int(x) for x in sorted(best)]
