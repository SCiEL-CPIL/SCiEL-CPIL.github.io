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
        # Generate all new sums
        new_sums = [s + r for s in sums_list]
        
        # Check for collisions using two pointers on the merged sorted lists
        # sums_list is sorted, new_sums is sorted
        i = 0
        j = 0
        last_val = None
        while i < n or j < n:
            if j >= n or (i < n and sums_list[i] < new_sums[j]):
                val = sums_list[i]
                i += 1
            elif i >= n or new_sums[j] < sums_list[i]:
                val = new_sums[j]
                j += 1
            else:
                # sums_list[i] == new_sums[j] -> collision
                return None
            
            if last_val is not None and val == last_val:
                return None
            last_val = val
        
        # No collision, merge the lists efficiently
        # We can just concatenate and sort if n is small, but since both are sorted,
        # a merge is O(n). However, Python's sort is fast in C.
        # For n up to ~2^15, a merge might be faster than sort due to no comparisons?
        # Actually, sort is O(n log n) but highly optimized.
        # Let's use a manual merge to be safe and deterministic, or just sort.
        # Given the size (up to 2^16), sort might be competitive.
        # But we already have the merged order logic. Let's just return the sorted union.
        # Wait, the loop above verified distinctness.
        # We need the actual list for the next step.
        # Merging two sorted lists is O(n).
        res = []
        i = 0
        j = 0
        while i < n or j < n:
            if j >= n or (i < n and sums_list[i] <= new_sums[j]):
                res.append(sums_list[i])
                i += 1
            else:
                res.append(new_sums[j])
                j += 1
        return res

    def backtrack(selected, remaining, sums_list, target_size):
        """Recursively try to add elements from remaining to selected to reach target_size."""
        nonlocal best_result, best_size
        if time.time() > deadline:
            return False
        
        current_size = len(selected)
        if current_size == target_size:
            if target_size > best_size:
                best_result = selected[:]
                best_size = target_size
            return True
        
        # Pruning: if even adding all remaining can't reach target_size, skip
        if current_size + len(remaining) < target_size:
            return False
        # Pruning: if even adding all remaining can't beat best_size, skip
        if current_size + len(remaining) <= best_size:
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
            if best_size >= target_size:
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
