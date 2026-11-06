"""Maximum independent set on the square-sum conflict graph via branch-and-bound with bitset pruning."""

# EVOLVE-BLOCK-START
import math

def _is_square(n):
    r = math.isqrt(n)
    return r * r == n

def _build_conflict_bitsets():
    """Return list of ints where bit j set in masks[i] means i and j conflict (i+j is a perfect square)."""
    N = 256
    masks = [0] * N
    for i in range(N):
        val = i + 1
        m = 0
        # diagonal: 2*val square?
        if _is_square(2 * val):
            m |= (1 << i)
        for j in range(i + 1, N):
            if _is_square(val + (j + 1)):
                m |= (1 << j)
        masks[i] = m
    return masks

def _greedy_independent_set(cands, masks):
    """Greedy: repeatedly pick a vertex with smallest degree among cands, remove it and its neighbors.
    Returns the count of selected vertices (an independent set size)."""
    count = 0
    remaining = cands
    while remaining:
        # pick vertex with min (degree within remaining)
        # iterate bits
        best_v = -1
        best_deg = 1 << 30
        temp = remaining
        while temp:
            low = temp & (-temp)
            v = low.bit_length() - 1
            deg = (masks[v] & remaining).bit_count()
            if deg < best_deg:
                best_deg = deg
                best_v = v
                if deg == 0:
                    break
            temp -= low
        count += 1
        remaining &= ~(masks[best_v] | (1 << best_v))
    return count

def _solve_max_independent_set(masks, N, time_budget=None):
    """Branch and bound for maximum independent set. Returns a list of selected vertex indices."""
    import time
    start = time.time()
    full = (1 << N) - 1

    # Precompute degree ordering: sort vertices by degree descending for branching
    order = sorted(range(N), key=lambda v: masks[v].bit_count(), reverse=True)
    # We'll branch on vertices in this order; but for B&B we need dynamic candidate set.
    # Use a recursive search over candidate set C (bitset).
    # best_solution stored as bitset for speed, converted at end.
    best = [0]  # bitset
    best_size = [0]

    # Upper bound function: greedy independent set size of C
    def upper_bound(C):
        # greedy: repeatedly pick vertex of min degree in C
        count = 0
        rem = C
        while rem:
            best_v = -1
            best_deg = 1 << 30
            temp = rem
            while temp:
                low = temp & (-temp)
                v = low.bit_length() - 1
                deg = (masks[v] & rem).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_v = v
                    if deg == 0:
                        break
                temp -= low
            count += 1
            rem &= ~(masks[best_v] | (1 << best_v))
        return count

    def search(C, current):
        import time as _t
        if _t.time() - start > time_budget:
            return
        # Upper bound: if |current| + ub(C) <= best_size, prune
        ub = upper_bound(C)
        if len(current) + ub <= best_size[0]:
            return
        if C == 0:
            if len(current) > best_size[0]:
                best_size[0] = len(current)
                best[0] = 0
                for v in current:
                    best[0] |= (1 << v)
            return

        # Pick a vertex to branch on: choose one with high degree in C for better pruning
        # But we want to include/exclude. Standard: pick v, branch include v, then exclude v (remove v and neighbors from C).
        # Choose v as vertex with max degree in C to maximize pruning in the exclude branch.
        temp = C
        v = -1
        maxdeg = -1
        while temp:
            low = temp & (-temp)
            idx = low.bit_length() - 1
            deg = (masks[idx] & C).bit_count()
            if deg > maxdeg:
                maxdeg = deg
                v = idx
            temp -= low

        # Branch 1: include v
        newC = C & ~(masks[v] | (1 << v))
        search(newC, current + [v])

        # Branch 2: exclude v
        C2 = C & ~(1 << v)
        search(C2, current)

    # Initial upper bound on full set
    search(full, [])

    # Convert best bitset to list of values (1-indexed)
    result = []
    b = best[0]
    while b:
        low = b & (-b)
        idx = low.bit_length() - 1
        result.append(idx + 1)
        b -= low
    return result

def construct():
    import time
    start = time.time()
    N = 256
    masks = _build_conflict_bitsets()
    # Use branch and bound with time budget
    time_budget = 55.0
    result = _solve_max_independent_set(masks, N, time_budget)
    # Fallback if something went wrong
    if not result:
        return list(range(1, 257, 3))
    # Validate: ensure it's a valid independent set
    res_set = set(result)
    for a in res_set:
        for b in res_set:
            if a <= b:
                if _is_square(a + b):
                    # invalid; fall back
                    return list(range(1, 257, 3))
    return sorted(res_set)
# EVOLVE-BLOCK-END
