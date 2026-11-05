"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def construct():
    """
    Return a nonempty subset of [1,22] such that ALL reciprocal subset sums,
    including the empty sum, are distinct. Maximize cardinality.
    """
    deadline = time.time() + 55  # Safety margin before 60s limit

    # LCM of 1..22
    D = 232792560
    vals = {x: D // x for x in range(1, 23)}

    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]  # Fallback size 9
    best_size = 9

    all_elements = list(range(1, 23))

    def sums_of_subset(subset):
        """Compute the set of all subset sums of vals[x] for x in subset."""
        sums = {0}
        for x in subset:
            v = vals[x]
            new_sums = set()
            for s in sums:
                new_sums.add(s + v)
            sums |= new_sums
        return sums

    def is_dissociated(subset):
        """Check if all subset sums are distinct (dissociated set)."""
        if len(subset) == 0:
            return True
        sums = sums_of_subset(subset)
        # If all subset sums are distinct, we have 2^k distinct sums
        return len(sums) == (1 << len(subset))

    # Try to find the largest valid set by checking sizes from high to low
    # We'll use a randomized search with pruning

    def try_find_set(target_size):
        """Try to find a valid set of the given size."""
        nonlocal best_set, best_size

        if target_size <= best_size:
            return False

        # Generate random permutations and try them
        for attempt in range(200):
            if time.time() > deadline:
                return False

            # Create a random permutation of all elements
            perm = all_elements[:]
            random.shuffle(perm)

            # Backtracking search
            current = []
            current_sums = {0}

            def backtrack(idx, remaining):
                nonlocal current, current_sums

                if time.time() > deadline:
                    return False

                if remaining == 0:
                    best_set = list(current)
                    best_size = len(current)
                    return True

                if len(perm) - idx < remaining:
                    return False

                for i in range(idx, len(perm)):
                    x = perm[i]
                    v = vals[x]

                    # Check if adding v creates a collision
                    # New sums would be current_sums | {s + v for s in current_sums}
                    # Collision if (s + v) in current_sums for any s in current_sums
                    # Equivalently: if any difference s1 - s2 = v exists in current_sums

                    valid = True
                    for s in current_sums:
                        if s + v in current_sums:
                            valid = False
                            break

                    if not valid:
                        continue

                    # Add x
                    current.append(x)
                    old_sums = current_sums
                    new_sums = set(old_sums)
                    for s in old_sums:
                        new_sums.add(s + v)
                    current_sums = new_sums

                    if backtrack(i + 1, remaining - 1):
                        return True

                    # Undo
                    current.pop()
                    current_sums = old_sums

                return False

            if backtrack(0, target_size):
                return True

        return False

    # Also try some heuristic-based approaches
    # Strategy: Start with large denominators (small values) and add greedily

    def greedy_search(target_size):
        """Greedy approach: add elements in order of increasing value (decreasing denominator)."""
        nonlocal best_set, best_size

        if target_size <= best_size:
            return False

        # Sort by value ascending (denominator descending) - smaller values are "safer"
        sorted_elems = sorted(all_elements, key=lambda x: vals[x])

        current = []
        current_sums = {0}

        for x in sorted_elems:
            if len(current) >= target_size:
                break

            v = vals[x]

            # Check if adding v creates a collision
            valid = True
            for s in current_sums:
                if s + v in current_sums:
                    valid = False
                    break

            if valid:
                current.append(x)
                old_sums = current_sums
                new_sums = set(old_sums)
                for s in old_sums:
                    new_sums.add(s + v)
                current_sums = new_sums

        if len(current) >= target_size and is_dissociated(current[:target_size]):
            best_set = current[:target_size]
            best_size = target_size
            return True

        return False

    # Try sizes from 17 down to 10
    for target_size in range(17, 9, -1):
        if target_size <= best_size:
            break

        if greedy_search(target_size):
            break

        if try_find_set(target_size):
            break

    # Final verification and return
    if best_size > 9:
        return best_set
    else:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
# EVOLVE-BLOCK-END
