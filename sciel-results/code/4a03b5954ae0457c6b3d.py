"""Reciprocal-subsum-free set over [1,24]. Maximize cardinality."""
import time
from math import prod
from itertools import combinations

_START = time.time()
_LIMIT = 80.0  # safety margin before 90s wall-clock kill

def _deadline():
    return time.time() - _START >= _LIMIT

# Scale reciprocals to integers: 1/x = LCM(1..24)/x, so subset-sum over integers is exact.
_LCM = 5354228880
def _w(x):
    return _LCM // x

# Precompute weights for all elements
_W = {x: _w(x) for x in range(1, 25)}

def _subset_sums_reach(target, weights):
    """Return True if some subset of weights sums to target (0 excluded)."""
    # DP bitset via set of reachable sums (bounded by target)
    reach = {0}
    for w in weights:
        if w > target:
            continue
        # add new sums
        new = set()
        for s in reach:
            ns = s + w
            if ns == target:
                return True
            if ns < target:
                new.add(ns)
        if new:
            reach |= new
    return False

def is_valid_set(S):
    """No reciprocal of a selected element equals sum of reciprocals of a subset of the OTHERS."""
    S = list(S)
    n = len(S)
    for i in range(n):
        target = _W[S[i]]
        others = [_W[S[j]] for j in range(n) if j != i]
        if _subset_sums_reach(target, others):
            return False
    return True

def can_add(S, x):
    """Check if adding x keeps the set valid (x not already in S)."""
    if x in S:
        return False
    S = list(S)
    # Check x's reciprocal not subsummed by others in S
    if _subset_sums_reach(_W[x], [_W[s] for s in S]):
        return False
    # Check each existing element's reciprocal not subsummed by (others + x)
    for i, s in enumerate(S):
        others = [_W[S[j]] for j in range(len(S)) if j != i]
        others.append(_W[x])
        if _subset_sums_reach(_W[s], others):
            return False
    return True

def greedy_construct(order):
    """Greedy add in given order if valid."""
    S = []
    for x in order:
        if can_add(S, x):
            S.append(x)
    return S

def local_add(S):
    """Try to add any element not in S that keeps validity; repeat until none."""
    S = list(S)
    changed = True
    while changed:
        changed = False
        for x in range(1, 25):
            if x not in S and can_add(S, x):
                S.append(x)
                changed = True
                break
    return S

def local_swap(S):
    """Try removing one and adding one/two to increase size; keep best."""
    S = list(S)
    best = S
    for x in list(S):
        T = [s for s in S if s != x]
        # try adding any two not in T
        for y in range(1, 25):
            if y in T:
                continue
            if can_add(T, y):
                U = T + [y]
                if len(U) > len(best):
                    best = U
        # try adding one
        for y in range(1, 25):
            if y in T:
                continue
            if can_add(T, y):
                U = T + [y]
                if len(U) > len(best):
                    best = U
    return best

def beam_search(width=8, max_depth=24):
    """Beam over subsets, growing one element at a time, pruning invalid."""
    # states: list of sorted tuples
    beams = [()]
    for step in range(max_depth):
        if _deadline():
            break
        cand = []
        seen = set()
        for state in beams:
            for x in range(1, 25):
                if x in state:
                    continue
                ns = tuple(sorted(state + (x,)))
                if ns in seen:
                    continue
                seen.add(ns)
                if can_add(list(state), x):
                    cand.append(ns)
        if not cand:
            break
        # score: prefer larger size, then smaller sum (looser)
        cand.sort(key=lambda t: (-len(t), sum(t)))
        beams = cand[:width]
        if _deadline():
            break
    best = max(beams, key=len) if beams else ()
    return list(best)

def backtrack_greedy(orders, beam_width=6):
    best = []
    for order in orders:
        if _deadline():
            break
        S = greedy_construct(order)
        S = local_add(S)
        if len(S) > len(best):
            best = S
        if _deadline():
            break
    return best

def construct():
    global _START, _LIMIT
    _START = time.time()
    _LIMIT = 80.0

    # Deterministic strong starting orders
    orders = [
        list(range(13, 25)),                 # seed
        list(range(1, 25)),                  # ascending
        list(range(24, 0, -1)),              # descending
        [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    ]

    best = []
    # 1) Greedy + local add over orders
    for order in orders:
        if _deadline():
            break
        S = greedy_construct(order)
        S = local_add(S)
        if len(S) > len(best):
            best = S
        if _deadline():
            break

    # 2) Beam search
    if not _deadline():
        B = beam_search(width=8, max_depth=24)
        B = local_add(B)
        if len(B) > len(best):
            best = B

    # 3) Local swap polish on best
    if not _deadline():
        P = local_swap(best)
        if len(P) > len(best):
            best = P

    # 4) Final local add polish
    if not _deadline():
        best = local_add(best)

    if not best:
        best = [13]

    return best
