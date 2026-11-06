"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
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

    # Try to find maximum cardinality set via backtracking
    # Order: try smaller indices first (larger reciprocals) for better pruning
    order = list(range(1, 23))

    best_result = [1, 2, 3, 5, 7, 11, 13, 17, 19]

    def backtrack(selected, remaining, sums, diffs):
        """Recursively try to add elements from remaining to selected.
        sums: set of all subset sums
        diffs: set of all pairwise differences (s1 - s2 for s1, s2 in sums)
        """
        nonlocal best_result
        if time.time() > deadline:
            return
        if len(selected) > len(best_result):
            best_result = selected[:]

        if not remaining:
            return

        n = len(remaining)
        for i in range(n):
            idx = remaining[i]
            r = recip[idx - 1]
            # Collision check: adding r is invalid if r is in diffs or -r is in diffs
            # because that means s1 + r = s2 for some existing sums s1, s2
            if r in diffs or (-r) in diffs:
                continue
            # Build new sums and diffs
            new_sums = set(sums)
            for s in sums:
                new_sums.add(s + r)
            # New diffs: old diffs + (s+r) - s' for all s, s' in sums
            # = old diffs + (s - s') + r and (s - s') - r
            new_diffs = set(diffs)
            for d in diffs:
                new_diffs.add(d + r)
                new_diffs.add(d - r)
            # Also need diffs involving the new sums vs old sums
            # (s+r) - s' = (s - s') + r, already covered
            # s - (s'+r) = (s - s') - r, already covered
            # But we also need the differences between the new sums themselves:
            # (s1+r) - (s2+r) = s1 - s2, which is in old diffs
            # And we need 0 - r and r - 0 = -r and r
            new_diffs.add(r)
            new_diffs.add(-r)

            selected.append(idx)
            backtrack(selected, remaining[i+1:], new_sums, new_diffs)
            selected.pop()
            # Early termination if we found a very large set
            if len(best_result) >= 16:
                return

    # Start backtracking with initial sums={0}, diffs={0}
    backtrack([], order, {0}, {0})

    return best_result
# EVOLVE-BLOCK-END
