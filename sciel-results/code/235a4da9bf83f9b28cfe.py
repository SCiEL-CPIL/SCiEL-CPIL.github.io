"""Two Sidon sets with disjoint difference sets - fresh design."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

N = 256
MAX_EACH = 64


def _pair_sums_ok(lst):
    """All pair sums (including a+a) in lst must be unique."""
    if len(lst) <= 1:
        return True
    seen = set()
    for i in range(len(lst)):
        a = lst[i]
        for j in range(i, len(lst)):
            s = a + lst[j]
            if s in seen:
                return False
            seen.add(s)
    return True


def _diffs(lst):
    """Set of nonzero differences b-a for distinct a,b in lst."""
    d = set()
    for i in range(len(lst)):
        a = lst[i]
        for j in range(i + 1, len(lst)):
            d.add(lst[j] - a)
    return d


def _can_add(lst, x, other_diffs):
    """Can we add x to lst keeping it Sidon and with disjoint differences from other_diffs?"""
    n = len(lst)
    # difference check: x - lst[i] and lst[i] - x must not be in other_diffs
    for i in range(n):
        a = lst[i]
        d1 = x - a
        d2 = a - x
        if d1 in other_diffs or d2 in other_diffs:
            return False
    # pair-sum uniqueness: for each i, x+lst[i] must not collide with any existing sum
    # existing sums include lst[i]+lst[j] for i<=j. We check against all pairs.
    for i in range(n):
        s = x + lst[i]
        for j in range(n):
            if lst[i] + lst[j] == s:
                # collision (note i==j gives lst[i]+lst[i], allowed as the same value? no)
                return False
    return True


def _greedy(order, deadline):
    """Greedy build two sets from a given candidate ordering."""
    sets = [[], []]
    diffs = [set(), set()]
    for x in order:
        if time.time() > deadline:
            break
        for side in (0, 1):
            if len(sets[side]) >= MAX_EACH:
                continue
            if _can_add(sets[side], x, diffs[1 - side]):
                sets[side].append(x)
                # update diffs for this side
                for a in sets[side][:-1]:
                    diffs[side].add(x - a)
                    diffs[side].add(a - x)
                break
    return sets


def _polish(sets, other_diffs, deadline):
    """Try to insert skipped elements via local search; returns improved set."""
    # For a single set with fixed other_diffs, try adding elements from [1..N]
    best = list(sets)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for x in range(1, N + 1):
            if x in best:
                continue
            if len(best) >= MAX_EACH:
                break
            if _can_add(best, x, other_diffs):
                best.append(x)
                for a in best[:-1]:
                    pass  # diffs recomputed by caller
                improved = True
    return best


def _full_polish(sets, deadline):
    """Iteratively polish both sets against each other's current differences."""
    A, B = list(sets[0]), list(sets[1])
    for _ in range(3):
        if time.time() > deadline:
            break
        dA = _diffs(A)
        A = _polish(A, _diffs(B), deadline)
        dB = _diffs(B)
        B = _polish(B, dA, deadline)
    return [A, B]


def _score(sets):
    """Validate and return total size, or 0 if invalid."""
    A, B = sets
    if not A or not B:
        return 0
    if len(A) > MAX_EACH or len(B) > MAX_EACH:
        return 0
    if not _pair_sums_ok(A) or not _pair_sums_ok(B):
        return 0
    dA = _diffs(A)
    dB = _diffs(B)
    if dA & dB:
        return 0
    return len(A) + len(B)


def _random_order():
    """Diverse candidate ordering: low-first with noise."""
    base = list(range(1, N + 1))
    # shuffle with a bias toward lower numbers
    order = list(base)
    random.shuffle(order)
    # sort by (value + noise) to keep low-first tendency
    order.sort(key=lambda x: x + random.random() * 40)
    return order


def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin within 90s
    best = [[1], [2]]
    best_score = 0

    # deterministic baseline
    sets = _greedy(list(range(1, N + 1)), deadline)
    sets = _full_polish(sets, deadline)
    sc = _score(sets)
    if sc > best_score:
        best_score = sc
        best = [list(sets[0]), list(sets[1])]

    # randomized restarts
    rng = random.Random(12345)
    while time.time() < deadline - 5.0:
        order = _random_order()
        sets = _greedy(order, deadline)
        sets = _full_polish(sets, deadline)
        sc = _score(sets)
        if sc > best_score:
            best_score = sc
            best = [list(sets[0]), list(sets[1])]

    # final safety polish on best
    if time.time() < deadline:
        best = _full_polish(best, deadline)
        sc = _score(best)
        if sc < best_score:
            # revert if polish broke validity
            best = [[1], [2]]
            best_score = 0

    if best_score == 0:
        return [[1], [2]]
    return best
# EVOLVE-BLOCK-END
