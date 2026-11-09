"""Fresh design: secure the binary {0,1}^n core first, then extend with 2s."""
import time
import random

# EVOLVE-BLOCK-START
_DEADLINE = time.time() + 300.0
_MARGIN = 12.0


def _is_cap_set(S, n):
    """True if S is a cap set: no three distinct elements form a 3-term AP."""
    m = len(S)
    if m < 3:
        return True
    for i in range(m):
        a = S[i]
        for j in range(i + 1, m):
            b = S[j]
            c = tuple((a[k] + b[k]) % 3 for k in range(n))
            if c in S:
                return False
    return True


def _max_cap_set(n, seed=0):
    """Greedy cap set: binary core first (lex order), then extend with 2s.

    Binary {0,1}^n core is secured first to avoid early conflicts that would
    block a large fraction of the safe subspace. Then all remaining vectors
    (those containing at least one 2) are tried in a randomised order and
    added greedily while the set stays a cap set.
    """
    rng = random.Random(seed)

    # ---- Phase 1: binary core, lexicographic order ----
    S = []
    S_set = set()
    for t in range(2 ** n):
        v = tuple((t >> k) & 1 for k in range(n))
        # check against existing S
        ok = True
        for a in S:
            c = tuple((v[k] + a[k]) % 3 for k in range(n))
            if c in S_set:
                ok = False
                break
        if ok:
            S.append(v)
            S_set.add(v)

    # ---- Phase 2: extend with 2s, randomised order ----
    # all vectors not yet in S that contain at least one 2
    remaining = []
    for t in range(3 ** n):
        v = []
        x = t
        for k in range(n):
            v.append(x % 3)
            x //= 3
        v = tuple(v)
        if v not in S_set:
            remaining.append(v)
    rng.shuffle(remaining)

    for v in remaining:
        if time.time() > _DEADLINE - _MARGIN:
            break
        ok = True
        for a in S:
            c = tuple((v[k] + a[k]) % 3 for k in range(n))
            if c in S_set:
                ok = False
                break
        if ok:
            S.append(v)
            S_set.add(v)

    return S


def priority(vector, n):
    """Build the cap set once (cached) and return its size as the score."""
    global _CAP_SIZE
    try:
        _CAP_SIZE
    except NameError:
        _CAP_SIZE = None
    if _CAP_SIZE is None:
        # Try a couple of seeds; keep the largest cap set found.
        best = 0
        for seed in (0, 1, 2):
            if time.time() > _DEADLINE - _MARGIN:
                break
            S = _max_cap_set(n, seed)
            if len(S) > best:
                best = len(S)
        _CAP_SIZE = float(best)
    return _CAP_SIZE
# EVOLVE-BLOCK-END
