import time
from math import lcm
from functools import reduce

def construct():
    start = time.time()
    deadline = start + 82.0

    L = reduce(lcm, range(1, 23))
    recip = {i: L // i for i in range(1, 23)}

    # Greedy initialization to find a large valid set quickly
    greedy_sums = {0}
    greedy_chosen = []
    for i in range(22, 0, -1):
        r = recip[i]
        # Fast collision check: (sums + r) & sums should be empty (except 0+r=r which is fine if r not in sums)
        shifted = {s + r for s in greedy_sums}
        if not (shifted & greedy_sums):
            greedy_chosen.append(i)
            greedy_sums = greedy_sums | shifted
    
    best = list(greedy_chosen)
    best_len = len(best)

    def can_add(sums, r):
        # collision if exists s1,s2 in sums with s1 - s2 == r
        # i.e., (sums + r) intersects sums
        shifted = {s + r for s in sums}
        return not (shifted & sums)

    # Dynamic variable ordering: at each step, pick the element that is "most constrained"
    # among remaining. A good heuristic: smaller reciprocal (larger i) is easier to place,
    # but we want to place harder-to-place elements first to prune early.
    # Actually, for dissociated sets, larger reciprocals (smaller i) are harder to fit.
    # So try smaller i first in the search to prune early.
    # But we also want to maximize cardinality, so we need balance.
    # Let's use a static ordering: try smaller i first (larger reciprocals) since they're
    # more likely to cause conflicts, so failing early prunes more.
    order = list(range(1, 23))

    # Precompute reciprocals in order
    recip_order = [recip[i] for i in order]

    def dfs(idx, sums, chosen):
        nonlocal best, best_len
        if time.time() > deadline:
            return
        # prune: even adding all remaining can't beat best
        remaining = len(order) - idx
        if len(chosen) + remaining <= best_len:
            return
        if idx == len(order):
            if len(chosen) > best_len:
                best = list(chosen)
                best_len = len(chosen)
            return
        
        i = order[idx]
        r = recip_order[idx]
        
        # Try include first (to find larger sets quickly)
        if can_add(sums, r):
            new_sums = sums | {s + r for s in sums}
            chosen.append(i)
            dfs(idx + 1, new_sums, chosen)
            chosen.pop()
        
        # Try exclude
        dfs(idx + 1, sums, chosen)

    dfs(0, {0}, [])
    return best
