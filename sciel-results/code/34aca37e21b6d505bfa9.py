"""Maximize cardinality of a dissociated reciprocal subset of [1,22].

Uses exact LCM integer arithmetic with backtracking and difference-set pruning.
Tries multiple element orderings to find the maximum cardinality set.
"""
import time
from math import lcm
from functools import reduce

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # LCM of 1..22 for exact integer representation
    L = reduce(lcm, range(1, 23))

    # Precompute reciprocal as integer: L // i
    recip = [L // i for i in range(1, 23)]

    # Known good baseline
    best_result = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_len = len(best_result)

    def backtrack(selected, remaining, sums, diffs):
        """Recursively try to add elements from remaining to selected."""
        nonlocal best_result, best_len
        if time.time() > deadline:
            return
        if len(selected) > best_len:
            best_result = selected[:]
            best_len = len(selected)

        if not remaining:
            return

        # Branch and bound: can't beat best if even adding all remaining won't help
        if len(selected) + len(remaining) <= best_len:
            return

        n = len(remaining)
        for i in range(n):
            idx = remaining[i]
            r = recip[idx - 1]
            # Collision check
            if r in diffs or (-r) in diffs:
                continue

            # Build new sums
            new_sums = set(sums)
            for s in sums:
                new_sums.add(s + r)

            # Build new diffs
            new_diffs = set(diffs)
            for d in diffs:
                new_diffs.add(d + r)
                new_diffs.add(d - r)
            new_diffs.add(r)
            new_diffs.add(-r)

            selected.append(idx)
            backtrack(selected, remaining[i+1:], new_sums, new_diffs)
            selected.pop()

            # Early termination if we found a very large set
            if best_len >= 18:
                return

    # Try descending order (larger indices first = smaller reciprocals)
    order_desc = list(range(22, 0, -1))
    backtrack([], order_desc, {0}, {0})

    # Try ascending order if we have time
    if time.time() < deadline - 5:
        order_asc = list(range(1, 23))
        backtrack([], order_asc, {0}, {0})

    # Try a mixed order: even indices first, then odd
    if time.time() < deadline - 5:
        order_mixed = [i for i in range(1, 23, 2)] + [i for i in range(2, 23, 2)]
        backtrack([], order_mixed, {0}, {0})

    if not best_result:
        best_result = [1]

    return best_result
