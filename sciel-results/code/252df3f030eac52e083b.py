"""Rewritten program: multi-ordering greedy + local-search on the 2-shadow conflict graph."""
# EVOLVE-BLOCK-START
import time
from math import gcd

# ---------- Precomputation ----------
def _build_conflict():
    """conflict[i][j] is True if i and j are in a bad triple (i,j,k) for some k != i,j."""
    n = 129
    L = [[0]*n for _ in range(n)]
    for i in range(1, n):
        for j in range(i+1, n):
            g = gcd(i, j)
            L[i][j] = (i//g) * j
    # For each pair (i,j), the set of k (k != i,j) that form a bad triple:
    # lcm(i,j) == lcm(i,k) == lcm(j,k) == L[i][j]
    # Build map L -> list of numbers whose pairwise lcm with others can equal L.
    # More directly: for each L value, find the set S_L = {x in [1,128] : x divides L and ... }
    # Actually lcm(i,j)=L[i][j]. A third k conflicts with pair (i,j) iff
    # lcm(i,k)==L[i][j] and lcm(j,k)==L[i][j].
    # We'll precompute, for each value v, the set of numbers that have lcm v with a given other.
    # Simpler: for each k, and each L, count how many of the pairs (i,j) with L[i][j]==L
    # also satisfy lcm(i,k)==L and lcm(j,k)==L.
    # Build: for each v, group indices by their "lcm-with-x" is complex. Instead:
    # For each triple check is O(n^3) = 2M, fine.
    conflict = [[False]*n for _ in range(n)]
    for i in range(1, n):
        for j in range(i+1, n):
            Lij = L[i][j]
            for k in range(1, n):
                if k == i or k == j:
                    continue
                if L[i][k] == Lij and L[j][k] == Lij:
                    conflict[i][j] = True
                    break
    return conflict, L

_CONFLICT, _L = _build_conflict()
_N = 129
_NUMS = list(range(1, _N))

# ---------- Greedy from a given ordering ----------
def _greedy(order, conflict):
    chosen = set()
    for x in order:
        ok = True
        for y in chosen:
            if conflict[x][y]:
                ok = False
                break
        if ok:
            chosen.add(x)
    return chosen

# ---------- Local search ----------
def _local_search(chosen, conflict):
    chosen = set(chosen)
    improved = True
    while improved:
        improved = False
        # Try adding any unchosen element that has no conflict with chosen
        for x in _NUMS:
            if x in chosen:
                continue
            ok = True
            for y in chosen:
                if conflict[x][y]:
                    ok = False
                    break
            if ok:
                chosen.add(x)
                improved = True
                break
        # Try swapping: remove one chosen, add one unchosen, net +1 (replace 1 with 2? not possible directly)
        # Try: for each unchosen x, and each chosen y, if removing y allows adding x and another z -> net +1
        # Simpler 1-for-1 swap won't increase size; do 2-for-1: remove two, add three? keep simple:
        # Try replacing a chosen y with unchosen x that conflicts with y but frees up: complex.
        # Do a "drop and re-greedy" around a random removal:
        for y in list(chosen):
            cand = chosen - {y}
            # re-add greedily in original order restricted
            for x in _NUMS:
                if x in cand:
                    continue
                ok = True
                for z in cand:
                    if conflict[x][z]:
                        ok = False
                        break
                if ok:
                    cand.add(x)
            if len(cand) > len(chosen):
                chosen = cand
                improved = True
                break
    return chosen

def construct():
    deadline = time.time() + 50.0
    conflict = _CONFLICT
    best = set()

    # Multiple orderings
    orders = [
        _NUMS,
        list(reversed(_NUMS)),
        sorted(_NUMS, key=lambda x: -x.bit_count()),
        sorted(_NUMS, key=lambda x: x.bit_count()),
        sorted(_NUMS, key=lambda x: (x % 2, x)),
        sorted(_NUMS, key=lambda x: (x.bit_length(), x)),
    ]
    for order in orders:
        if time.time() > deadline:
            break
        cand = _greedy(order, conflict)
        cand = _local_search(cand, conflict)
        if len(cand) > len(best):
            best = cand

    # Random restarts
    import random
    rng = random.Random(12345)
    while time.time() < deadline:
        order = list(_NUMS)
        rng.shuffle(order)
        cand = _greedy(order, conflict)
        cand = _local_search(cand, conflict)
        if len(cand) > len(best):
            best = cand

    if not best:
        best = {1}
    return list(best)
# EVOLVE-BLOCK-END
