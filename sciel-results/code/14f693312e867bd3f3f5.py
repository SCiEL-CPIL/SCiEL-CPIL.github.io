"""Rewritten program: multi-greedy + local search for equal-gcd triple avoidance."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

_TIME_LIMIT = 84.0  # safety margin before the 90s wall-clock kill
_DEADLINE = None


def _is_valid_triple_free(S):
    """Return True if no three distinct elements have all three pairwise gcds equal."""
    n = len(S)
    # Precompute gcd for pairs lazily; check triples
    # For each pair (i,j), gcd_ij. For a third k, need gcd(i,k)==gcd(j,k)==gcd(i,j)
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            g = gcd(a, b)
            for k in range(j + 1, n):
                c = S[k]
                if gcd(a, c) == g and gcd(b, c) == g:
                    return False
    return True


def _can_add(S, x, gcd_cache):
    """Check if adding x to S keeps the set triple-free. gcd_cache maps (u,v)->gcd."""
    for a in S:
        ga = gcd(a, x)
        for b in S:
            if b == a:
                continue
            gb = gcd(b, x)
            if ga == gb:
                g_ab = gcd(a, b)
                if g_ab == ga:
                    return False
    return True


def _greedy(available, order, time_deadline):
    """Greedy: iterate 'order', add if valid. Returns list."""
    S = []
    Sset = set()
    for x in order:
        if time.time() > time_deadline:
            break
        if x in Sset:
            continue
        if _can_add(S, x, None):
            S.append(x)
            Sset.add(x)
    return S


def _local_search(S, time_deadline, rng):
    """Try to grow S by adding elements and removing one at a time (2-opt style)."""
    S = list(S)
    Sset = set(S)
    universe = list(range(1, 129))
    improved = True
    while improved:
        if time.time() > time_deadline:
            break
        improved = False
        # Try adding any element not in S
        for x in universe:
            if time.time() > time_deadline:
                break
            if x in Sset:
                continue
            if _can_add(S, x, None):
                S.append(x)
                Sset.add(x)
                improved = True
                break
        if not improved:
            # Try 2-opt: remove one, then try to add two (or one better)
            # Simpler: try removing each element and see if a new one fits, net gain
            for idx in range(len(S)):
                if time.time() > time_deadline:
                    break
                removed = S[idx]
                S2 = S[:idx] + S[idx+1:]
                # try to add something
                best_add = None
                for x in universe:
                    if x in S2:
                        continue
                    if _can_add(S2, x, None):
                        best_add = x
                        break
                if best_add is not None:
                    # net zero size; only accept if it opens further gains — skip for simplicity
                    pass
    return S


def _multi_greedy_local(time_deadline):
    rng = random.Random(12345)
    universe = list(range(1, 129))
    best = []

    # Deterministic orders first
    orders = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
        list(range(1, 129, 2)) + list(range(2, 129, 2)),
        [x for x in range(1, 129) if x % 2 == 0] + [x for x in range(1, 129) if x % 2 == 1],
    ]
    # Add some random orders
    for _ in range(8):
        o = universe[:]
        rng.shuffle(o)
        orders.append(o)

    for order in orders:
        if time.time() > time_deadline:
            break
        cand = _greedy(universe, order, time_deadline)
        if len(cand) > len(best):
            best = cand

    # Local search on best
    best = _local_search(best, time_deadline, rng)

    # Try local search on a few other good candidates
    for order in orders[:4]:
        if time.time() > time_deadline:
            break
        cand = _greedy(universe, order, time_deadline)
        cand = _local_search(cand, time_deadline, rng)
        if len(cand) > len(best):
            best = cand

    return best


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT
    try:
        result = _multi_greedy_local(_DEADLINE)
    except Exception:
        # Fallback: simple ascending greedy
        result = []
        for x in range(1, 129):
            ok = True
            for a in result:
                ga = gcd(a, x)
                for b in result:
                    if b == a:
                        continue
                    if ga == gcd(b, x) == gcd(a, b):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                result.append(x)
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
