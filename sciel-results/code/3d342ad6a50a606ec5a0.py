import time
import random
from fractions import Fraction

_START = time.time()
_DEADLINE = _START + 80.0  # safety margin before 90s kill

_valid_cache = {}

def _can_sum_to(others, target):
    """True if some nonempty subset of `others` sums exactly to `target` (exact Fractions)."""
    sums = {Fraction(0)}
    for r in others:
        if r > target:
            continue
        new = set()
        for s in sums:
            t = s + r
            if t <= target:
                new.add(t)
        sums |= new
        if target in sums:
            return True
    return False

def _valid(S):
    """S valid iff for every x in S, 1/x is not a subset-sum of {1/y : y in S, y != x}."""
    key = frozenset(S)
    if key in _valid_cache:
        return _valid_cache[key]
    recip = {x: Fraction(1, x) for x in S}
    ok = True
    for x in S:
        target = recip[x]
        others = [recip[y] for y in S if y != x]
        if _can_sum_to(others, target):
            ok = False
            break
    _valid_cache[key] = ok
    return ok

def _greedy(order):
    S = []
    for x in order:
        if time.time() > _DEADLINE:
            break
        if _valid(S + [x]):
            S.append(x)
    return S

def construct():
    all_elems = list(range(1, 25))
    best = list(range(13, 25))  # known-valid fallback
    best_size = len(best)

    def consider(S):
        nonlocal best, best_size
        if S and _valid(S) and len(S) > best_size:
            best_size = len(S)
            best = list(S)

    # --- Phase 1: greedy from several orderings ---
    orders = [sorted(all_elems, reverse=True), sorted(all_elems)]
    for _ in range(30):
        o = all_elems[:]
        random.shuffle(o)
        orders.append(o)

    frontier = []
    for o in orders:
        if time.time() > _DEADLINE:
            break
        S = _greedy(o)
        consider(S)
        if len(S) >= best_size - 1:
            frontier.append(frozenset(S))

    # --- Phase 2: improve each frontier set by adding excluded elements ---
    for f in frontier:
        if time.time() > _DEADLINE:
            break
        S = list(f)
        comp = [x for x in all_elems if x not in S]
        improved = True
        while improved and time.time() < _DEADLINE:
            improved = False
            for x in sorted(comp, reverse=True):
                if _valid(S + [x]):
                    S.append(x)
                    comp.remove(x)
                    improved = True
                    consider(S)

        # --- Phase 3: light 2-for-1 swaps to push cardinality ---
        if time.time() < _DEADLINE:
            comp2 = [x for x in all_elems if x not in S]
            for a in list(S):
                if time.time() > _DEADLINE:
                    break
                S_no_a = [y for y in S if y != a]
                for b in comp2:
                    S1 = S_no_a + [b]
                    if _valid(S1):
                        for c in [x for x in all_elems if x not in S1]:
                            S2 = S1 + [c]
                            if _valid(S2):
                                consider(S2)
                                S = S2
                                comp2 = [x for x in all_elems if x not in S]
                                break

    # --- Final safety: guarantee a valid nonempty answer ---
    if not best or not _valid(best):
        best = list(range(13, 25))
    return best
