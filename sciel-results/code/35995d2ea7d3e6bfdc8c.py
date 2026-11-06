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
        shifted = {s + r for s in greedy_sums}
        if not (shifted & greedy_sums):
            greedy_chosen.append(i)
            greedy_sums = greedy_sums | shifted

    best = list(greedy_chosen)
    best_len = len(best)

    def can_add(sums, r):
        shifted = {s + r for s in sums}
        return not (shifted & sums)

    # Order: smaller i first (larger reciprocals) since they're more constraining
    order = list(range(1, 23))
    recip_order = [recip[i] for i in order]
    n = len(order)

    # Precompute suffix counts for pruning
    # remaining_count[idx] = number of elements from idx to n-1
    remaining_count = [0] * (n + 1)
    for idx in range(n - 1, -1, -1):
        remaining_count[idx] = remaining_count[idx + 1] + 1

    def dfs(idx, sums, chosen):
        nonlocal best, best_len
        if time.time() > deadline:
            return
        # Prune: even adding all remaining can't beat best
        if len(chosen) + remaining_count[idx] <= best_len:
            return
        if idx == n:
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
