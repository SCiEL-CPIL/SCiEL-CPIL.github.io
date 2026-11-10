"""Rewritten program: branch-and-bound over {1..22} for max dissociated reciprocal set."""
import time
from math import gcd
from functools import reduce

def _lcm_all(lst):
    return reduce(lambda a, b: a * b // gcd(a, b), lst, 1)

def construct():
    start_time = time.time()
    deadline = start_time + 75  # safety margin under 90s limit

    universe = list(range(1, 23))
    N = len(universe)

    # Precompute integer reciprocals: 1/e = (L//e)/L  where L = lcm(1..22)
    L = _lcm_all(universe)
    num = [L // e for e in universe]  # integer numerator for each reciprocal

    # Order candidates to make search efficient:
    # Larger elements have smaller reciprocals (smaller numerator) -> fewer collisions.
    # Search in descending value order.
    order = list(range(N - 1, -1, -1))  # indices for elements 22,21,...,1
    vals = [universe[i] for i in order]
    nums = [num[i] for i in order]

    best = []
    best_len = 0

    # Branch and bound with a set of reachable subset-sum numerators.
    # A new element with numerator n collides iff n is in the difference set,
    # equivalently iff (s + n) in S for some s in S (where S = current subset sums).
    # We maintain S as a set of ints.

    def _time_ok():
        return (time.time() - start_time) < deadline

    # Iterative DFS with explicit stack to allow time checks and pruning.
    # State: (pos, current_list, current_sums)
    # We try to include element at 'pos' then exclude.

    # Use recursion with manual time guard via a global counter.
    import sys
    sys.setrecursionlimit(10000)

    def dfs(pos, cur_list, cur_sums):
        nonlocal best, best_len

        # Prune: even if we add all remaining, can't beat best
        remaining = N - pos
        if len(cur_list) + remaining <= best_len:
            return
        if not _time_ok():
            return

        if pos == N:
            if len(cur_list) > best_len:
                best_len = len(cur_list)
                best = cur_list[:]
            return

        # Option 1: include element at pos
        n = nums[pos]
        # Check collision: does adding n create a duplicate?
        # New sums = cur_sums | {s + n for s in cur_sums}
        # Collision if any (s + n) in cur_sums
        collision = False
        for s in cur_sums:
            if (s + n) in cur_sums:
                collision = True
                break
        if not collision:
            new_sums = cur_sums | {s + n for s in cur_sums}
            cur_list.append(vals[pos])
            dfs(pos + 1, cur_list, new_sums)
            cur_list.pop()

        if not _time_ok():
            return

        # Option 2: exclude element at pos
        dfs(pos + 1, cur_list, cur_sums)

    dfs(0, [], {0})

    # Convert best (values) to sorted list
    if best:
        return sorted(best)

    # Fallback: a known dissociated set
    return [1, 2, 3, 5, 7, 11, 13, 17, 19]
