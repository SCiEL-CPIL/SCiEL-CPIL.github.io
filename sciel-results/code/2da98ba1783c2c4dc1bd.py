"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    # LCM of 1..22 for exact integer representation
    L = reduce(lcm, range(1, 23))

    # Precompute reciprocal as integer: L // i
    recip = [L // i for i in range(1, 23)]

    # Order elements by reciprocal value descending (larger reciprocals first)
    # This helps with pruning: larger elements create more distinct sums faster
    order = sorted(range(1, 23), key=lambda x: -recip[x - 1])

    best_result = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_result)

    def try_add(sums_list, r):
        """Try to add r to the sorted list of sums. Returns new list if valid, None if collision."""
        n = len(sums_list)
        # Generate all new sums: s + r for s in sums_list
        # Check for collisions: no two sums in (sums_list ∪ new_sums) can be equal
        # Since sums_list is sorted, new_sums = [s+r for s in sums_list] is also sorted
        # We need to check that the merged list has no duplicates
        # Efficient: merge two sorted lists and check for duplicates
        i = 0
        j = 0
        last = None
        while i < n or j < n:
            if j >= n or (i < n and sums_list[i] < sums_list[j] + r):
                val = sums_list[i]
                i += 1
            elif i >= n or sums_list[j] + r < sums_list[i]:
                val = sums_list[j] + r
                j += 1
            else:
                # sums_list[i] == sums_list[j] + r → collision
                return None
            if last is not None and val == last:
                return None
            last = val
        # Build new sorted list
        new_list = []
        i = 0
        j = 0
        while i < n or j < n:
            if j >= n or (i < n and sums_list[i] <= sums_list[j] + r):
                new_list.append(sums_list[i])
                i += 1
            else:
                new_list.append(sums_list[j] + r)
                j += 1
        return new_list

    def backtrack(selected, remaining, sums_list, target_size):
        """Recursively try to add elements from remaining to selected to reach target_size."""
        nonlocal best_result, best_size
        if time.time() > deadline:
            return False
        if len(selected) == target_size:
            if target_size > best_size:
                best_result = selected[:]
                best_size = target_size
            return True
        # Upper bound: can we reach target_size?
        if len(selected) + len(remaining) < target_size:
            return False
        # Pruning: if even adding all remaining can't beat best, skip
        if len(selected) + len(remaining) <= best_size:
            return False

        for i in range(len(remaining)):
            idx = remaining[i]
            r = recip[idx - 1]
            new_sums = try_add(sums_list, r)
            if new_sums is not None:
                selected.append(idx)
                if backtrack(selected, remaining[i+1:], new_sums, target_size):
                    selected.pop()
                    return True
                selected.pop()
            # Early termination if we found a solution of this target size
            if len(best_result) >= target_size:
                return True
        return False

    # Try to find maximum cardinality, starting from high and going down
    # We know at least 9 works. Try 16, 15, 14, ...
    for target in range(16, 8, -1):
        if time.time() > deadline:
            break
        found = backtrack([], order, [0], target)
        if found:
            break

    return best_result
# EVOLVE-BLOCK-END
