"""Branch-and-bound search for closest strict underpacking of 1 by unit fractions 1/2..1/48."""
# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _build_tables():
    from math import lcm
    L = 1
    for d in range(2, 49):
        L = lcm(L, d)
    n = 47
    vals = [L // d for d in range(2, 49)]
    suffix = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix[i] = suffix[i + 1] + vals[i]
    suffix_float = [0.0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_float[i] = suffix_float[i + 1] + 1.0 / (i + 2)
    return L, vals, suffix, suffix_float

_L, _VALS, _SUFFIX, _SUFFIX_FLOAT = _build_tables()
_N = 47

def construct():
    start = time.time()
    deadline = start + 85.0  # safety margin under 90s wall-clock
    L = _L
    vals = _VALS
    suffix = _SUFFIX
    suffix_float = _SUFFIX_FLOAT
    N = _N

    # Greedy warm start: include every item that keeps sum strictly below 1.
    greedy_mask = 0
    cur = 0
    for i in range(N):
        if cur + vals[i] < L:
            greedy_mask |= (1 << i)
            cur += vals[i]
    best_deficit = L - cur
    best_mask = greedy_mask

    # Branch-and-bound over subsets.
    # State: (i, cur_int, mask). cur_int is exact integer sum of chosen items.
    # Admissible upper bound on final sum = cur_int + suffix[i].
    # Prune if cur_int + suffix[i] <= L - best_deficit (can't beat best)
    # and prune if cur_int + suffix[i] < L is impossible to improve.
    # We want to maximize cur_int subject to cur_int < L.

    def dfs(i, cur_int, mask):
        nonlocal best_deficit, best_mask
        if i == N:
            d = L - cur_int
            if d < best_deficit:
                best_deficit = d
                best_mask = mask
            return
        # Fast float prune: even taking everything remaining can't beat best deficit
        if cur_int + suffix[i] <= L - best_deficit:
            return
        # Branch: try taking item i (prefer taking to reduce deficit)
        # Take
        nxt = cur_int + vals[i]
        if nxt < L:
            dfs(i + 1, nxt, mask | (1 << i))
            if time.time() >= deadline:
                return
        # Skip
        dfs(i + 1, cur_int, mask)

    dfs(0, 0, 0)

    # Reconstruct indices from mask
    indices = [i for i in range(N) if (best_mask >> i) & 1]
    if not indices:
        # Fallback: any single item
        indices = [0]
    return indices
# EVOLVE-BLOCK-END
