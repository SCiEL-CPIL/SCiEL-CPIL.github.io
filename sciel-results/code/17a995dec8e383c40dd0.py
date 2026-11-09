"""Combined solver: maximize cardinality of S ⊆ [1,192] (|S|≤48) such that
no selected integer divides the sum of any 3 distinct OTHER selected integers.

Structural idea (inspiration 2): fast O(n^2) per-element remainder-counting
validity check. Refinement machinery (island-1 improvement): safety-scored
greedy over multiple orderings plus swap-based local search to grow the set.
"""

import time
import random
from collections import defaultdict

# Deadline guard: evaluator kills after 90s wall-clock; keep a safety margin.
_DEADLINE = 82.0
_START = time.time()


def _time_left():
    return _DEADLINE - (time.time() - _START)


def _is_safe_element(a, others):
    """Return True iff element `a` does NOT divide the sum of any 3 distinct
    elements drawn from `others` (a list of distinct positive ints, a not in it)."""
    L = others
    m = len(L)
    if m < 3:
        return True
    # Count remainders of each element of L modulo a.
    rem_count = defaultdict(int)
    for v in L:
        rem_count[v % a] += 1
    for i in range(m):
        u = L[i]
        for j in range(i + 1, m):
            v = L[j]
            target = (-u - v) % a
            cnt = rem_count.get(target, 0)
            # Exclude u and v themselves as the third element.
            if u % a == target:
                cnt -= 1
            if v % a == target:
                cnt -= 1
            if cnt > 0:
                return False
    return True


def _is_valid(S):
    """Full validity check for a list of distinct positive integers."""
    n = len(S)
    if n < 4:
        return True
    for idx, a in enumerate(S):
        others = [S[k] for k in range(n) if k != idx]
        if not _is_safe_element(a, others):
            return False
    return True


def _safety_score(x, S):
    """Estimate how 'safe' it is to add x to S (lower is safer).

    Counts, over every pair of existing elements, the number of distinct
    existing elements that would complete a bad triple for a given divisor.
    """
    score = 0
    n = len(S)
    # 1) x as the tested element: existing triples whose sum is divisible by x.
    for i in range(n):
        u = S[i]
        for j in range(i + 1, n):
            v = S[j]
            target = (-u - v) % x
            cnt = 0
            for k in range(n):
                if k != i and k != j and S[k] % x == target:
                    cnt += 1
            score += cnt
    # 2) each existing element as tested element with x as a new summand.
    for idx, a in enumerate(S):
        for i in range(n):
            if i == idx:
                continue
            u = S[i]
            target = (-x - u) % a
            cnt = 0
            for k in range(n):
                if k != idx and k != i and S[k] % a == target:
                    cnt += 1
            score += cnt
    return score


def _greedy_with_safety(order, S):
    """Extend S with as many elements as possible from `order`, choosing at each
    step the candidate with the lowest safety score (ties broken by larger value)."""
    S = list(S)
    S_set = set(S)
    for x in order:
        if len(S) >= 48:
            break
        if _time_left() < 2.0:
            break
        if x in S_set:
            continue
        new_S = S + [x]
        if not _is_valid(new_S):
            continue
        S.append(x)
        S_set.add(x)
    return S


def _local_search(S):
    """Try to grow S by swapping one element out for one in, then greedily
    filling remaining capacity. Keeps the best valid set found."""
    best = list(S)
    best_len = len(best)
    best_set = set(best)
    n = 192
    attempts = 0
    while _time_left() > 2.0 and len(best) < 48:
        attempts += 1
        if attempts % 200 == 0 and _time_left() < 3.0:
            break
        # Random removal candidate
        out_idx = random.randrange(len(best))
        out_val = best[out_idx]
        remaining = [best[k] for k in range(len(best)) if k != out_idx]
        rem_set = set(remaining)
        # Candidate additions not already in the set
        candidates = [x for x in range(1, n + 1) if x not in rem_set and x != out_val]
        # Try a few candidates, prefer those that keep validity and grow the set.
        found = False
        for x in candidates:
            if len(best) >= 48:
                break
            trial = remaining + [x]
            if not _is_valid(trial):
                continue
            # Greedily fill remaining capacity from the leftover pool.
            trial_set = set(trial)
            pool = [y for y in range(1, n + 1) if y not in trial_set]
            trial = _greedy_with_safety(pool, trial)
            if len(trial) > best_len:
                best = trial
                best_len = len(best)
                best_set = set(best)
                found = True
                break
        if found:
            continue
        # No improvement this round; keep going (bounded by time).
    return best


def construct():
    n = 192
    best = []
    best_len = 0

    # --- Pass A: greedy from largest to smallest (baseline, matches seed) ---
    S = []
    S_set = set()
    for x in range(n, 0, -1):
        if len(S) >= 48:
            break
        if _time_left() < 3.0:
            break
        new_S = S + [x]
        if _is_valid(new_S):
            S.append(x)
            S_set.add(x)
    if len(S) > best_len:
        best = list(S)
        best_len = len(S)

    # --- Pass B: safety-scored greedy over several orderings ---
    rng = random.Random(12345)
    orderings = []
    orderings.append(list(range(n, 0, -1)))
    orderings.append(list(range(1, n + 1)))
    orderings.append(list(range(n, 0, -2)))   # large evens
    orderings.append(list(range(1, n + 1, 2)))  # small odds
    for _ in range(6):
        o = list(range(1, n + 1))
        rng.shuffle(o)
        orderings.append(o)

    for order in orderings:
        if _time_left() < 4.0:
            break
        S = _greedy_with_safety(order, [])
        if len(S) > best_len:
            best = list(S)
            best_len = len(S)

    # --- Pass C: local search from the current best to push toward 48 ---
    if _time_left() > 5.0:
        improved = _local_search(best)
        if len(improved) > best_len:
            best = improved
            best_len = len(best)

    # Final safety: ensure validity before returning.
    if best and not _is_valid(best):
        # Rebuild a guaranteed-valid baseline quickly.
        S = []
        for x in range(n, 0, -1):
            if len(S) >= 48:
                break
            if not _is_valid(S + [x]):
                continue
            S.append(x)
        best = S

    return list(best)
