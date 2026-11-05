import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 90s wall-clock, leave 5s safety margin

    max_val = 128

    # Precompute LCM table
    lcm_table = [[0] * (max_val + 1) for _ in range(max_val + 1)]
    for i in range(1, max_val + 1):
        for j in range(i, max_val + 1):
            lcm_table[i][j] = i * j // gcd(i, j)
            lcm_table[j][i] = lcm_table[i][j]

    # Build conflict structure: for each pair (a,b), the set of c's that form a conflicting triple
    # A triple (a,b,c) with a<b<c is conflicting if lcm(a,b)==lcm(a,c)==lcm(b,c)
    conflict_map = {}  # (min,max) -> set of other elements that form conflict with this pair

    for a in range(1, max_val + 1):
        for b in range(a + 1, max_val + 1):
            l_ab = lcm_table[a][b]
            for c in range(b + 1, max_val + 1):
                if lcm_table[a][c] == l_ab and lcm_table[b][c] == l_ab:
                    # (a,b,c) is a conflicting triple
                    if (a, b) not in conflict_map:
                        conflict_map[(a, b)] = set()
                    conflict_map[(a, b)].add(c)
                    if (a, c) not in conflict_map:
                        conflict_map[(a, c)] = set()
                    conflict_map[(a, c)].add(b)
                    if (b, c) not in conflict_map:
                        conflict_map[(b, c)] = set()
                        conflict_map[(b, c)].add(a)

    # Also build a "conflict partners" structure for faster checking
    # For each element x, and each other element y, if (x,y) is in conflict_map,
    # then adding x to a set containing y requires that none of conflict_map[(x,y)] are in the set.

    def check_add(x, current_set):
        """Check if x can be added to current_set without creating a conflict."""
        for a in current_set:
            key = (a, x) if a < x else (x, a)
            if key in conflict_map:
                for c in conflict_map[key]:
                    if c in current_set:
                        return False
        return True

    def greedy_construct(order):
        """Build a valid set by greedily adding elements in the given order."""
        current_set = set()
        for x in order:
            if check_add(x, current_set):
                current_set.add(x)
        return current_set

    def try_add_all(current_set, not_in_set):
        """Try to add any element from not_in_set that can be added. Returns modified sets."""
        improved = True
        while improved:
            improved = False
            for x in list(not_in_set):
                if check_add(x, current_set):
                    current_set.add(x)
                    not_in_set.remove(x)
                    improved = True
        return current_set, not_in_set

    # Strategy 1: Greedy with various orderings
    # Sort by conflict count (fewer conflicts = safer)
    conflict_count = [0] * (max_val + 1)
    for (a, b), cs in conflict_map.items():
        for c in cs:
            conflict_count[a] += 1
            conflict_count[b] += 1
            conflict_count[c] += 1

    best_set = set()
    best_size = 0

    # Try sorted by conflict count
    order = list(range(1, max_val + 1))
    order.sort(key=lambda x: conflict_count[x])
    s = greedy_construct(order)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s

    # Try reverse order
    order.sort(key=lambda x: -conflict_count[x])
    s = greedy_construct(order)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s

    # Try natural order
    order = list(range(1, max_val + 1))
    s = greedy_construct(order)
    if len(s) > best_size:
        best_size = len(s)
        best_set = s

    # Try random orderings
    for _ in range(50):
        if time.time() > deadline - 10:
            break
        random.shuffle(order)
        s = greedy_construct(order)
        if len(s) > best_size:
            best_size = len(s)
            best_set = s

    # Polish: try to add remaining elements
    current_set = set(best_set)
    not_in_set = set(range(1, max_val + 1)) - current_set
    current_set, not_in_set = try_add_all(current_set, not_in_set)
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set

    # Strategy 2: Local search with swaps
    # Try to remove one element and add two or more
    for iteration in range(10):
        if time.time() > deadline - 5:
            break

        current_set = set(best_set)
        not_in_set = set(range(1, max_val + 1)) - current_set
        improved = False

        # Try each element in current_set as a removal candidate
        for x in list(current_set):
            if time.time() > deadline - 5:
                break
            # Temporarily remove x
            current_set.remove(x)
            not_in_set.add(x)

            # Try to add elements from not_in_set
            added = []
            for y in list(not_in_set):
                if y == x:
                    continue
                if check_add(y, current_set):
                    current_set.add(y)
                    not_in_set.remove(y)
                    added.append(y)

            # If we added more than we removed, keep the change
            if len(added) > 1:
                improved = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set
                break  # Restart with the new set
            else:
                # Revert
                current_set.add(x)
                not_in_set.remove(x)
                for y in added:
                    current_set.remove(y)
                    not_in_set.add(y)

        if not improved:
            break

    # Final polish
    current_set = set(best_set)
    not_in_set = set(range(1, max_val + 1)) - current_set
    current_set, not_in_set = try_add_all(current_set, not_in_set)
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set

    # Strategy 3: Simulated annealing-style search
    # Start from best_set and try random swaps
    if time.time() < deadline - 5:
        current_set = set(best_set)
        not_in_set = set(range(1, max_val + 1)) - current_set

        for _ in range(200):
            if time.time() > deadline - 3:
                break

            # Randomly pick an element to remove and try to add others
            if not current_set:
                break
            x = random.choice(list(current_set))
            current_set.remove(x)
            not_in_set.add(x)

            # Try to add elements
            added = []
            for y in random.sample(list(not_in_set), min(len(not_in_set), 10)):
                if y == x:
                    continue
                if check_add(y, current_set):
                    current_set.add(y)
                    not_in_set.remove(y)
                    added.append(y)

            if len(added) > 1 and len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set
            else:
                # Revert
                current_set.add(x)
                not_in_set.remove(x)
                for y in added:
                    current_set.remove(y)
                    not_in_set.add(y)

    # Final polish
    current_set = set(best_set)
    not_in_set = set(range(1, max_val + 1)) - current_set
    current_set, not_in_set = try_add_all(current_set, not_in_set)
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set

    return list(best_set)
