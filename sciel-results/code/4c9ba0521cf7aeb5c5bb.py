from fractions import Fraction
from itertools import combinations

def _build_dangerous():
    targets = {1: Fraction(1,1), 2: Fraction(1,2), 3: Fraction(1,3), 4: Fraction(1,4)}
    nums = list(range(1,25))
    dangerous = {a: [] for a in targets}
    def dfs(a_target, start, remaining, current):
        if remaining == 0:
            if current:
                dangerous[a_target].append(tuple(current))
            return
        if remaining < 0:
            return
        for i in range(start, 24):
            v = nums[i]
            if v == a_target:
                continue
            frac = Fraction(1, v)
            if frac > remaining:
                continue
            # upper bound: sum of 1/(i+1)+...+1/24
            # simple prune
            current.append(v)
            dfs(a_target, i+1, remaining - frac, current)
            current.pop()
    for a, tgt in targets.items():
        dfs(a, 0, tgt, [])
    return dangerous

_DANGEROUS = _build_dangerous()

def _is_valid(S):
    Sset = set(S)
    for a in (1,2,3,4):
        if a in Sset:
            for D in _DANGEROUS[a]:
                # D is tuple, doesn't contain a
                if all(x in Sset for x in D):
                    return False
    return True

def construct():
    full = list(range(1,25))
    # Try sizes 24,23,22
    for k in (24,23,22):
        for combo in combinations(full, k):
            if _is_valid(combo):
                return list(combo)
    # Fallback: find any valid size 21 (e.g., {4..24})
    cand = list(range(4,25))
    assert _is_valid(cand)
    return cand
