"""Rewritten program: large sum-free-ish set in 1..128 with no bad triple."""
import time

_N = 128
_SUM_MAX = 2 * _N  # 256
_BIT_LIMIT = 1 << _SUM_MAX  # 1 << 256

def _has_bad_triple(S):
    """Return True iff there exist distinct a<b<c in S with all three pair sums in S."""
    s = set(S)
    n = len(S)
    if n < 3:
        return False
    for i in range(n):
        a = S[i]
        for j in range(i + 1, n):
            b = S[j]
            for k in range(j + 1, n):
                c = S[k]
                if (a + b) in s and (a + c) in s and (b + c) in s:
                    return True
    return False

def _validate(S):
    """Full validity check: distinct ints in 1..N, no bad triple."""
    if len(S) != len(set(S)):
        return False
    for x in S:
        if not isinstance(x, int) or isinstance(x, bool):
            return False
        if x < 1 or x > _N:
            return False
    return not _has_bad_triple(S)

def _greedy_random(seed, time_budget):
    """Randomized greedy: try numbers in a seeded random order (descending bias),
    add if it creates no bad triple with the current set. Returns best set."""
    import random
    rng = random.Random(seed)
    # Bias toward larger numbers (they are less likely to form sums with others in-range?
    # Actually larger numbers have larger pair sums, less likely to be in 1..128,
    # so they are "safer". Descending greedy is a good heuristic.)
    candidates = list(range(_N, 0, -1))
    rng.shuffle(candidates)
    # But to bias large, we can do a partial: keep top half in random order then bottom
    # Simpler: just shuffle fully; large numbers naturally appear and are safe.
    S = []
    s_set = set()
    start = time.time()
    for x in candidates:
        if time.time() - start > time_budget:
            break
        # Check: does adding x create a bad triple?
        # A bad triple involving x: either x is one of a,b,c, or x is one of the sums.
        # Case 1: x is one of the three members.
        #   Then we need two other members y,z in S such that the three pair sums
        #   are all in S ∪ {x}.
        # Case 2: x is one of the pair sums.
        #   Then there exist a,b in S with a+b = x, and we need a+c, b+c in S∪{x}
        #   for some c in S.
        # This is complex; for small N=128, just do the O(n^3) check on S∪{x}.
        # Since n will be ~60-70, n^3 ~ 300k, times 128 candidates = 40M ops. Might be slow.
        # Use a faster check.
        S.append(x)
        s_set.add(x)
        if _has_bad_triple(S):
            S.pop()
            s_set.discard(x)
    return S

def _remove_polish(S, time_budget):
    """Try to remove elements to fix any remaining issues, or try adding back removed ones."""
    # First, if S is valid, try to add back elements we might have missed.
    s_set = set(S)
    S = list(S)
    start = time.time()
    improved = True
    while improved and time.time() - start < time_budget:
        improved = False
        for x in range(_N, 0, -1):
            if time.time() - start > time_budget:
                break
            if x in s_set:
                continue
            S.append(x)
            s_set.add(x)
            if _has_bad_triple(S):
                S.pop()
                s_set.discard(x)
            else:
                improved = True
    return S

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin before 90s kill

    best = list(range(_N // 2 + 1, _N + 1))  # seed: 65..128, size 64
    best_size = len(best)

    # Try multiple random seeds
    seed = 0
    while time.time() < deadline - 5.0:
        budget = min(10.0, deadline - time.time() - 5.0)
        if budget <= 0:
            break
        S = _greedy_random(seed, budget)
        if len(S) > best_size:
            best = S
            best_size = len(S)
        seed += 1

    # Polish the best
    if time.time() < deadline - 3.0:
        best = _remove_polish(best, deadline - time.time() - 3.0)

    # Final validation
    if not _validate(best):
        # Fallback to seed
        best = list(range(_N // 2 + 1, _N + 1))

    return list(best)
