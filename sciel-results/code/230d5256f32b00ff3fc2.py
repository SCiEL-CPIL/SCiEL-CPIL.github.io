"""Search-based program: varied candidate build + local search, keep best within budget."""
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
_BASE = 51975
_DEADLINE = 140.0          # safety margin before the 150 s kill
_START = time.time()
_rng = random.Random(20240517)

def _divisors(n):
    divs = []
    for d in range(2, n + 1):
        if n % d == 0:
            divs.append(d)
    return divs

_DIVS = _divisors(_BASE)          # 41 distinct odd divisors, > 1
# pool of small-to-large moduli used as the search focus
_POOL = [d for d in _DIVS if d <= 2835]

def _score(classes):
    """Exact uncovered count (0 = covered, 1 = uncovered)."""
    covered = bytearray(_BASE)
    for m, r in classes:
        covered[r::m] = b'\x01' * ((len(covered) - 1 - r) // m + 1)
    return _BASE - sum(covered)

def _build(perm, rand_res, t_start):
    """Greedy build over a given modulus permutation; returns (classes, score)."""
    covered = bytearray(_BASE)
    classes = []
    for m in perm:
        if rand_res:
            r = _rng.randrange(m)
        else:
            best_r = 0
            best_c = -1
            for r in range(m):
                c = sum(1 for x in covered[r::m] if x == 0)
                if c > best_c:
                    best_c = c
                    best_r = r
                    if best_c == 0:
                        break
            r = best_r
        classes.append((m, r))
        covered[r::m] = b'\x01' * ((len(covered) - 1 - r) // m + 1)
    return classes, _BASE - sum(covered)

def _local_search(classes, t_start):
    """Iterate: for each class, pick the residue minimizing uncovered count."""
    classes = [list(c) for c in classes]
    improved = True
    best_sc = _BASE
    while improved and (time.time() - t_start) < _DEADLINE - 5:
        improved = False
        for i, (m, r) in enumerate(classes):
            others = [(mm, rr) for j, (mm, rr) in enumerate(classes) if j != i]
            best_r = r
            best_sc_local = _BASE
            for nr in range(m):
                covered = bytearray(_BASE)
                for mm, rr in others:
                    covered[rr::mm] = b'\x01' * ((len(covered) - 1 - rr) // mm + 1)
                covered[nr::m] = b'\x01' * ((len(covered) - 1 - nr) // m + 1)
                sc = _BASE - sum(covered)
                if sc < best_sc_local:
                    best_sc_local = sc
                    best_r = nr
            if best_r != r:
                classes[i] = [m, best_r]
                improved = True
            if best_sc_local < best_sc:
                best_sc = best_sc_local
    return classes, best_sc

def construct():
    t_start = time.time()
    base = sorted(_POOL)          # smallest moduli first

    # deterministic strong base: smallest-modulus-first greedy
    classes, sc = _build(base, False, t_start)
    classes, sc = _local_search(classes, t_start)
    best = [c for c in classes]
    best_sc = sc

    # randomized restarts + local search
    while (time.time() - t_start) < _DEADLINE - 10:
        perm = _POOL[:]
        _rng.shuffle(perm)
        # bias: keep the very smallest moduli early sometimes
        classes, sc = _build(perm, True, t_start)
        if sc < best_sc:
            classes, sc = _local_search(classes, t_start)
            if sc < best_sc:
                best_sc = sc
                best = [c for c in classes]

    return [[m, r] for m, r in best]
# EVOLVE-BLOCK-END
