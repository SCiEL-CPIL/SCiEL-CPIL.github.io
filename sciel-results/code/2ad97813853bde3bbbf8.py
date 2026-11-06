"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import lcm
from functools import reduce

def construct():
    """
    Find a nonempty subset of [1,22] with maximum cardinality such that
    all reciprocal subset sums (including empty sum 0) are distinct.
    Uses meet-in-the-middle: partition into halves, precompute valid configs,
    then greedily combine while checking sum-set disjointness.
    """
    deadline = time.time() + 82  # safety margin before 90s kill

    # Compute LCM of 1..22 to convert reciprocal sums to integer sums
    L = reduce(lcm, range(1, 23))

    # For each element a in [1,22], its contribution to the integer sum is L // a
    vals = {a: L // a for a in range(1, 23)}

    def compute_subset_sums(elements_list):
        """Compute all subset sums for a list of elements (using vals)."""
        sums = {0}
        for a in elements_list:
            v = vals[a]
            new_sums = set()
            for s in sums:
                new_sums.add(s + v)
            sums = sums | new_sums
        return sums

    def is_dissociated(elements_list):
        """Check if all subset sums are distinct."""
        if not elements_list:
            return True
        sums = {0}
        for a in elements_list:
            v = vals[a]
            new_sums = set()
            for s in sums:
                ns = s + v
                if ns in sums or ns in new_sums:
                    return False
                new_sums.add(ns)
            sums = sums | new_sums
        return True

    def sums_disjoint(set1, set2):
        """Check if two sum sets are disjoint (no common elements)."""
        return len(set1 & set2) == 0

    # Meet-in-the-middle: partition [1,22] into two halves
    left_half = list(range(1, 12))   # 1-11
    right_half = list(range(12, 23)) # 12-22

    # Precompute all dissociated subsets of each half with their sum sets
    # For each half, enumerate all subsets and keep only dissociated ones
    def enumerate_dissociated_subsets(half_elements):
        """Return list of (subset_list, sum_set) for all dissociated subsets."""
        results = [([], {0})]  # empty subset
        n = len(half_elements)
        for mask in range(1, 1 << n):
            if time.time() > deadline - 5:
                break
            subset = []
            for i in range(n):
                if mask & (1 << i):
                    subset.append(half_elements[i])
            if is_dissociated(subset):
                sums = compute_subset_sums(subset)
                results.append((subset, sums))
        return results

    # Enumerate dissociated subsets for each half
    left_configs = enumerate_dissociated_subsets(left_half)
    right_configs = enumerate_dissociated_subsets(right_half)

    # Sort by cardinality descending, then by sum-set size ascending (smaller = more compatible)
    left_configs.sort(key=lambda x: (-len(x[0]), len(x[1])))
    right_configs.sort(key=lambda x: (-len(x[0]), len(x[1])))

    best_subset = []
    best_size = 0

    # Greedy combination: for each left config, find best compatible right config
    for left_subset, left_sums in left_configs:
        if time.time() > deadline - 3:
            break
        for right_subset, right_sums in right_configs:
            if time.time() > deadline - 3:
                break
            # Check if sum sets are disjoint
            if sums_disjoint(left_sums, right_sums):
                combined_size = len(left_subset) + len(right_subset)
                if combined_size > best_size:
                    best_size = combined_size
                    best_subset = left_subset + right_subset
                    # Early termination if we found maximum possible
                    if best_size == 22:
                        return best_subset
            # If right config is too large to beat current best, skip remaining
            if len(right_subset) + len(left_subset) <= best_size and len(right_configs) > 1:
                # Since sorted by size desc, if this one can't beat, neither can later ones
                # But we need to check if there's a smaller right that still combines well
                # Actually since sorted desc, all subsequent are <= this size
                break

    # Also try the reverse: iterate right first, then left
    for right_subset, right_sums in right_configs:
        if time.time() > deadline - 3:
            break
        for left_subset, left_sums in left_configs:
            if time.time() > deadline - 3:
                break
            if sums_disjoint(left_sums, right_sums):
                combined_size = len(left_subset) + len(right_subset)
                if combined_size > best_size:
                    best_size = combined_size
                    best_subset = left_subset + right_subset
                    if best_size == 22:
                        return best_subset
            if len(right_subset) + len(left_subset) <= best_size and len(left_configs) > 1:
                break

    # Fallback: simple greedy if meet-in-middle didn't find anything good
    if best_size < 1:
        elements = sorted(range(1, 23), key=lambda a: vals[a], reverse=True)
        current = []
        current_sums = {0}
        for a in elements:
            v = vals[a]
            new_sums = set()
            conflict = False
            for s in current_sums:
                ns = s + v
                if ns in current_sums or ns in new_sums:
                    conflict = True
                    break
                new_sums.add(ns)
            if not conflict:
                current.append(a)
                current_sums = current_sums | new_sums
        best_subset = current
        best_size = len(best_subset)

    if not best_subset:
        best_subset = [1]

    return best_subset

# EVOLVE-BLOCK-END
