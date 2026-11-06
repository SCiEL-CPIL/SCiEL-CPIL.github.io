import time
import random

def construct():
    start_time = time.time()
    n = 256
    deadline = start_time + 80.0  # Safety margin before 90s kill

    # Pre-compute all 3-APs and 4-APs
    # 3-AP: (a, b, c) where a < b < c and b-a == c-b
    # 4-AP: (a, b, c, d) where a < b < c < d and b-a == c-b == d-c

    # For each element, store which 3-APs and 4-APs it participates in
    # 3-AP: for element x, it can be:
    #   - first: (x, x+d, x+2d) for d >= 1, x+2d <= n
    #   - middle: (x-d, x, x+d) for d >= 1, x-d >= 1
    #   - last: (x-2d, x-d, x) for d >= 1, x-2d >= 1

    # 4-AP: for element x, it can be:
    #   - first: (x, x+d, x+2d, x+3d)
    #   - second: (x-d, x, x+d, x+2d)
    #   - third: (x-2d, x-d, x, x+d)
    #   - fourth: (x-3d, x-2d, x-d, x)

    # Build element -> list of (ap_id, role) for 3-APs and 4-APs
    # We'll use a dict: elem -> list of 3-AP tuples, elem -> list of 4-AP tuples

    three_aps = {}  # elem -> list of (a, b, c)
    four_aps = {}   # elem -> list of (a, b, c, d)

    for x in range(1, n + 1):
        three_aps[x] = []
        four_aps[x] = []

    # Generate all 3-APs
    for d in range(1, n // 2 + 1):
        for a in range(1, n - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            ap3 = (a, b, c)
            three_aps[a].append(ap3)
            three_aps[b].append(ap3)
            three_aps[c].append(ap3)

    # Generate all 4-APs
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            dd = a + 3 * d
            ap4 = (a, b, c, dd)
            four_aps[a].append(ap4)
            four_aps[b].append(ap4)
            four_aps[c].append(ap4)
            four_aps[dd].append(ap4)

    def count_three_aps(selected_set):
        """Count distinct 3-APs in the selected set."""
        count = 0
        for d in range(1, n // 2 + 1):
            for a in range(1, n - 2 * d + 1):
                b = a + d
                c = a + 2 * d
                if a in selected_set and b in selected_set and c in selected_set:
                    count += 1
        return count

    def has_four_ap(selected_set, x):
        """Check if adding x to selected_set would create a 4-AP."""
        for ap4 in four_aps[x]:
            a, b, c, d = ap4
            # Check if the other three elements are already in selected_set
            others = [elem for elem in ap4 if elem != x]
            if all(elem in selected_set for elem in others):
                return True
        return False

    # Greedy construction: try to build a set with many 3-APs and no 4-APs
    # Strategy: start with a base set, then greedily add elements that maximize 3-APs
    # while maintaining the no-4-AP constraint

    best_set = set()
    best_score = 0

    # Try multiple random seeds for diversity
    random.seed(42)

    for trial in range(100):
        if time.time() > deadline - 5.0:
            break

        # Start with a random subset or use a structured approach
        # Approach 1: Start with elements that form many 3-APs
        # Approach 2: Greedy addition

        selected = set()

        # Greedy: add elements in order of how many 3-APs they contribute
        # while checking 4-AP constraint

        # First, let's try a simple greedy: add elements that don't create 4-APs
        # and maximize 3-AP count

        # Order elements by potential 3-AP contribution
        elem_potential = {}
        for x in range(1, n + 1):
            # Count how many 3-APs x can be part of
            potential = len(three_aps[x])
            elem_potential[x] = potential

        # Try different ordering strategies
        if trial % 3 == 0:
            # Natural order
            order = list(range(1, n + 1))
        elif trial % 3 == 1:
            # By potential (descending)
            order = sorted(range(1, n + 1), key=lambda x: -elem_potential[x])
        else:
            # Random order
            order = list(range(1, n + 1))
            random.shuffle(order)

        for x in order:
            if time.time() > deadline - 2.0:
                break

            # Check if adding x creates a 4-AP
            if has_four_ap(selected, x):
                continue

            # Check if adding x is beneficial (creates new 3-APs)
            selected.add(x)

        # Count 3-APs in this set
        score = count_three_aps(selected)

        if score > best_score:
            best_score = score
            best_set = set(selected)

        # Local improvement: try to swap elements to improve 3-AP count
        if time.time() > deadline - 3.0:
            break

        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            for x in list(best_set):
                for y in range(1, n + 1):
                    if y in best_set:
                        continue
                    if time.time() > deadline - 1.0:
                        break

                    # Try swapping x for y
                    new_set = set(best_set)
                    new_set.remove(x)
                    new_set.add(y)

                    # Check 4-AP constraint
                    if has_four_ap(new_set, y):
                        continue

                    new_score = count_three_aps(new_set)
                    if new_score > best_score:
                        best_score = new_score
                        best_set = new_set
                        improved = True
                        break

                if time.time() > deadline - 1.0:
                    break

    # Final verification: ensure no 4-APs
    for ap4 in set(tuple(ap) for elems in four_aps.values() for ap in elems for ap4 in [ap]):
        if all(elem in best_set for elem in ap4):
            # Remove one element to break the 4-AP
            # This shouldn't happen if our constraint checking is correct
            pass

    return sorted(best_set) if best_set else [1]
# EVOLVE-BLOCK-END
