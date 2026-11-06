import time
from fractions import Fraction

N = 256
ALL = tuple(range(1, N + 1))

def construct():
    start = time.time()
    DEADLINE = start + 82.0   # 8s safety margin under the 90s kill

    # ---- Precompute conflict bitmasks for every element ----
    # conflict_mask[x] has bit y set iff {x, y} participates in a 3-AP (i.e. there
    # exists z such that x,y,z form a nonconstant 3-AP).  Adding x to S is valid
    # iff (S_mask & conflict_mask[x]) == 0.
    conflict_mask = [0] * (N + 1)
    for x in range(1, N + 1):
        m = 0
        # x as start: x, x+d, x+2d
        d = 1
        while x + 2 * d <= N:
            m |= 1 << (x + d)
            m |= 1 << (x + 2 * d)
            d += 1
        # x as middle: x-d, x, x+d
        d = 1
        while x - d >= 1 and x + d <= N:
            m |= 1 << (x - d)
            m |= 1 << (x + d)
            d += 1
        # x as end: x-2d, x-d, x
        d = 1
        while x - 2 * d >= 1:
            m |= 1 << (x - d)
            m |= 1 << (x - 2 * d)
            d += 1
        conflict_mask[x] = m

    inv = [None] + [Fraction(1, x) for x in range(1, N + 1)]

    # Candidates ordered by descending 1/x (small numbers first) -> tight early bound.
    order = list(range(1, N + 1))
    order.sort(key=lambda x: inv[x], reverse=True)
    order_idx = [0] * (N + 1)
    for i, x in enumerate(order):
        order_idx[x] = i

    # Suffix sums of 1/x over `order` for a fast admissible upper bound.
    # suffix[i] = sum_{j>=i} inv[order[j]]
    suffix = [Fraction(0)] * (len(order) + 1)
    for i in range(len(order) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + inv[order[i]]

    best_mask = 0
    best_sum = Fraction(0)

    # Greedy seed (small-first, keep only if addable) to give a strong incumbent.
    seed_mask = 0
    for x in order:
        if seed_mask & conflict_mask[x] == 0:
            seed_mask |= 1 << x
    seed_sum = Fraction(0)
    for x in order:
        if (seed_mask >> x) & 1:
            seed_sum += inv[x]
    best_mask = seed_mask
    best_sum = seed_sum

    # ---- Recursive Branch & Bound ----
    # State: current_mask, current_sum, pos (index into `order` of next candidate).
    # We decide for each candidate in `order` (small-first): include or skip.
    # Bound: current_sum + sum of 1/x over addable candidates from pos onward.
    # To keep the bound admissible yet fast, we bound by current_sum + suffix[pos]
    # (ignoring conflicts) -- cheap; plus a tighter addable-bound when cheap.

    import sys
    sys.setrecursionlimit(100000)

    nodes = [0]
    use_exact_bound = [True]

    def bound_addable(current_mask, pos):
        # sum of 1/x for candidates order[pos:] that are individually addable.
        s = Fraction(0)
        for j in range(pos, len(order)):
            x = order[j]
            if current_mask & conflict_mask[x] == 0:
                s += inv[x]
        return s

    def search(current_mask, current_sum, pos):
        nonlocal best_mask, best_sum
        if time.time() > DEADLINE:
            return
        nodes[0] += 1
        # Cheap optimistic bound (ignore conflicts): current_sum + suffix[pos].
        if current_sum + suffix[pos] <= best_sum:
            return
        if pos >= len(order):
            return
        x = order[pos]
        # Include x if addable.
        if current_mask & conflict_mask[x] == 0:
            new_sum = current_sum + inv[x]
            if new_sum > best_sum:
                # tentative incumbent (valid set)
                best_mask = current_mask | (1 << x)
                best_sum = new_sum
            # Recurse with x included.
            search(current_mask | (1 << x), new_sum, pos + 1)
            if time.time() > DEADLINE:
                return
        # Skip x.
        # Prune skip branch with a tighter addable bound if we have time budget.
        if use_exact_bound[0]:
            # addable bound for the skip branch (x not added)
            if current_sum + bound_addable(current_mask, pos + 1) <= best_sum:
                return
        search(current_mask, current_sum, pos + 1)

    search(seed_mask, seed_sum, 0)

    # ---- Final polish: local search (single swaps) on the best set ----
    def can_add_mask(x, mask):
        return (mask & conflict_mask[x]) == 0

    def set_sum(mask):
        s = Fraction(0)
        for x in range(1, N + 1):
            if (mask >> x) & 1:
                s += inv[x]
        return s

    cur_mask = best_mask
    cur_sum = best_sum
    improved = True
    while improved and time.time() < DEADLINE:
        improved = False
        in_set = [x for x in range(1, N + 1) if (cur_mask >> x) & 1]
        not_set = [x for x in range(1, N + 1) if not ((cur_mask >> x) & 1)]
        # low value first for removal, high value first for addition
        in_set.sort(key=lambda x: inv[x])
        not_set.sort(key=lambda x: inv[x], reverse=True)
        for u in in_set:
            if time.time() > DEADLINE:
                break
            base = cur_mask & ~(1 << u)
            for v in not_set:
                if inv[v] <= inv[u]:
                    break
                if can_add_mask(v, base):
                    ns = cur_sum - inv[u] + inv[v]
                    if ns > cur_sum:
                        cur_mask = base | (1 << v)
                        cur_sum = ns
                        improved = True
                        break
        if improved:
            if cur_sum > best_sum:
                best_sum = cur_sum
                best_mask = cur_mask
            # recompute lists next loop
            continue

    # Build result
    result = [x for x in range(1, N + 1) if (best_mask >> x) & 1]
    if not result:
        result = [1]
    return sorted(result)
