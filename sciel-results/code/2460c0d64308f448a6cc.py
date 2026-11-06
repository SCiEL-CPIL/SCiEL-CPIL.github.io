"""Maximum independent set on the square-sum conflict graph via branch-and-bound with bitset pruning."""

# EVOLVE-BLOCK-START
import math
import time

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

def _greedy_is_size(cands, masks):
    """Fast greedy independent set size on candidate bitset cands."""
    count = 0
    rem = cands
    while rem:
        # pick vertex with minimum degree in rem
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
            temp ^= low
        count += 1
        rem &= ~(masks[best_v] | (1 << best_v))
    return count

def _solve_max_independent_set(masks, N, time_budget=55.0):
    """Branch and bound for maximum independent set using bitsets.
    Returns list of selected vertex indices (0-based)."""
    start = time.time()
    full = (1 << N) - 1

    # Precompute static degrees for tie-breaking
    static_deg = [masks[v].bit_count() for v in range(N)]

    best_bitset = [0]
    best_size = [0]

    # Greedy initial solution to get a good lower bound
    init_sol = []
    rem = full
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
            temp ^= low
        init_sol.append(best_v)
        rem &= ~(masks[best_v] | (1 << best_v))
    best_size[0] = len(init_sol)
    b = 0
    for v in init_sol:
        b |= (1 << v)
    best_bitset[0] = b

    def upper_bound(C):
        """Greedy upper bound: size of greedy independent set on C."""
        return _greedy_is_size(C, masks)

    def search(C, cur_size, cur_bitset):
        nonlocal best_bitset, best_size
        if time.time() - start > time_budget:
            return
        if C == 0:
            if cur_size > best_size[0]:
                best_size[0] = cur_size
                best_bitset[0] = cur_bitset
            return
        # Upper bound prune
        ub = upper_bound(C)
        if cur_size + ub <= best_size[0]:
            return
        # Choose branching vertex: max degree in C, tie-break by static degree
        temp = C
        v = -1
        maxdeg = -1
        maxstatic = -1
        while temp:
            low = temp & (-temp)
            idx = low.bit_length() - 1
            deg = (masks[idx] & C).bit_count()
            if deg > maxdeg or (deg == maxdeg and static_deg[idx] > maxstatic):
                maxdeg = deg
                maxstatic = static_deg[idx]
                v = idx
            temp ^= low
        # Branch 1: include v
        newC = C & ~(masks[v] | (1 << v))
        search(newC, cur_size + 1, cur_bitset | (1 << v))
        # Branch 2: exclude v
        C2 = C & ~(1 << v)
        search(C2, cur_size, cur_bitset)

    search(full, 0, 0)

    # Convert best bitset to list of indices
    result = []
    b = best_bitset[0]
    while b:
        low = b & (-b)
        idx = low.bit_length() - 1
        result.append(idx)
        b ^= low
    return result

def construct():
    N = 256
    masks = _build_conflict_bitsets()
    time_budget = 55.0
    result_indices = _solve_max_independent_set(masks, N, time_budget)
    if not result_indices:
        return list(range(1, 257, 3))
    result = sorted(i + 1 for i in result_indices)
    # Validate
    res_set = set(result)
    for a in res_set:
        for b in res_set:
            if a <= b:
                if _is_square(a + b):
                    return list(range(1, 257, 3))
    return result
# EVOLVE-BLOCK-END
