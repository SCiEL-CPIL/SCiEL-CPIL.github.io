import time
import random

_DEADLINE = time.monotonic() + 85.0  # safety margin before 90s kill

def _valid(S):
    """Full validity check: for all a<b<=c in S, a does not divide b+c."""
    S = sorted(S)
    k = len(S)
    for i in range(k):
        a = S[i]
        for j in range(i + 1, k):
            b = S[j]
            for c in S[j:]:
                if (b + c) % a == 0:
                    return False
    return True

def _can_add(S, Sset, x):
    """Return True if S U {x} is valid, given S is valid. Only checks triples involving x."""
    # x as the smallest (a): for b,c in S with b<=c, x | (b+c)
    for b in S:
        if b <= x:
            continue
        for c in S:
            if c < b:
                continue
            if (b + c) % x == 0:
                return False
    # x as the middle (b): for a<x in S, c>=x in S, a | (x+c)
    for a in S:
        if a >= x:
            continue
        for c in S:
            if c < x:
                continue
            if (x + c) % a == 0:
                return False
    # x as the largest (c): for a<b<x in S, a | (b+x)
    for i in range(len(S)):
        a = S[i]
        for j in range(i + 1, len(S)):
            b = S[j]
            if b >= x:
                break
            if (b + x) % a == 0:
                return False
    return True

def _greedy_add(S, Sset, order, deadline):
    """Add as many elements from order as possible (in that order)."""
    S = list(S)
    Sset = set(S)
    for x in order:
        if time.monotonic() > deadline:
            break
        if x in Sset:
            continue
        if _can_add(S, Sset, x):
            S.append(x)
            Sset.add(x)
    S.sort()
    return S, Sset

def _improve(S0, deadline, rng):
    """Hill-climb with kicks: add feasible elements; when stuck, remove one and re-add."""
    S = sorted(S0)
    Sset = set(S)
    best = list(S)
    bestlen = len(S)
    candidates = list(range(1, 257))
    while time.monotonic() < deadline:
        # addition pass
        added = False
        for x in candidates:
            if time.monotonic() > deadline:
                break
            if x in Sset:
                continue
            if _can_add(S, Sset, x):
                S.append(x)
                Sset.add(x)
                S.sort()
                added = True
                break
        if added:
            if len(S) > bestlen:
                best = list(S)
                bestlen = len(S)
            continue
        # stuck: kick by removing a random element, then try to add
        if len(S) > 1:
            e = S[rng.randrange(len(S))]
            S.remove(e)
            Sset.discard(e)
            # try to add several after the kick
            for x in candidates:
                if time.monotonic() > deadline:
                    break
                if x in Sset:
                    continue
                if _can_add(S, Sset, x):
                    S.append(x)
                    Sset.add(x)
                    S.sort()
            if len(S) > bestlen:
                best = list(S)
                bestlen = len(S)
            # if the kick shrank us a lot and we can't recover, revert to best
            if len(S) < bestlen - 3:
                S = list(best)
                Sset = set(S)
    return best

def _sweep_remove_top_add_bottom(deadline):
    """Remove top r from [171,256], greedily add lower elements; track best."""
    base = list(range(171, 257))
    best = list(base)
    bestlen = len(base)
    for r in range(0, 40):
        if time.monotonic() > deadline:
            break
        S = base[r:]  # remove top r
        Sset = set(S)
        # add lower elements in descending order (larger lower elements are easier to add)
        lower = list(range(170 - r, 0, -1))
        S, Sset = _greedy_add(S, Sset, lower, deadline)
        if len(S) > bestlen:
            best = list(S)
            bestlen = len(S)
    return best

def construct():
    global _DEADLINE
    deadline = _DEADLINE
    rng = random.Random(12345)
    base = list(range(171, 257))  # size 86, valid
    best = list(base)
    bestlen = len(base)

    # Deterministic sweep first (cheap, structured)
    cand = _sweep_remove_top_add_bottom(deadline)
    if len(cand) > bestlen:
        best = cand
        bestlen = len(cand)

    # Multi-start local search with kicks
    starts = [
        list(range(171, 257)),
        list(range(172, 257)),
        list(range(175, 257)),
        list(range(180, 257)),
        list(range(160, 257)),  # invalid start; will be repaired by kicks? no, start must be valid
    ]
    # ensure starts are valid; [160,257) is invalid, drop it
    starts = [s for s in starts if _valid(s)]

    for s0 in starts:
        if time.monotonic() > deadline:
            break
        res = _improve(s0, deadline, rng)
        if len(res) > bestlen:
            best = res
            bestlen = len(res)

    # A few randomized restarts from perturbed valid sets
    for _ in range(200):
        if time.monotonic() > deadline:
            break
        s0 = list(best)
        # random perturbation: remove a few, then improve
        for _ in range(rng.randrange(1, 5)):
            if len(s0) > 2:
                e = s0[rng.randrange(len(s0))]
                s0.remove(e)
        res = _improve(s0, deadline, rng)
        if len(res) > bestlen:
            best = res
            bestlen = len(res)

    # Final validity guarantee
    if not _valid(best) or len(best) == 0:
        best = list(range(171, 257))
    return list(best)
