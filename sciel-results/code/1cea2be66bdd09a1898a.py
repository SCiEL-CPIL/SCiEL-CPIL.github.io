import time
import random
from math import gcd
from collections import defaultdict

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

    # Build adjacency list for the conflict graph
    # Two elements x,y are adjacent if there exists z such that (x,y,z) is a conflicting triple
    # i.e., lcm(x,y)==lcm(x,z)==lcm(y,z)
    # But wait - we need to be more careful. The constraint is:
    # No three distinct elements a,b,c with lcm(a,b)==lcm(a,c)==lcm(b,c)
    # This is NOT a pairwise constraint - it's a 3-uniform hypergraph constraint.
    # So we can't directly use MIS on a graph.
    #
    # Instead, let's think of it differently. For a set S to be valid, for every triple
    # (a,b,c) in S, we must NOT have lcm(a,b)==lcm(a,c)==lcm(b,c).
    #
    # Let's build a hypergraph where hyperedges are the conflicting triples,
    # and we want the maximum subset with no hyperedge fully contained.
    # This is the Maximum Independent Set in a 3-uniform hypergraph.
    #
    # For now, let's use a branch-and-bound approach on the hypergraph.

    # Build the list of all conflicting triples
    conflict_triples = []
    for (a, b), cs in conflict_map.items():
        for c in cs:
            if a < b < c:
                conflict_triples.append((a, b, c))
            elif a < c < b:
                conflict_triples.append((a, c, b))
            elif b < a < c:
                conflict_triples.append((b, a, c))
            elif b < c < a:
                conflict_triples.append((b, c, a))
            elif c < a < b:
                conflict_triples.append((c, a, b))
            elif c < b < a:
                conflict_triples.append((c, b, a))

    # Remove duplicates
    conflict_triples = list(set(conflict_triples))
    conflict_triples.sort()

    # For each element, store which triples it's in
    element_to_triples = defaultdict(set)
    for idx, (a, b, c) in enumerate(conflict_triples):
        element_to_triples[a].add(idx)
        element_to_triples[b].add(idx)
        element_to_triples[c].add(idx)

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

    # Strategy 2: Branch and Bound on the hypergraph
    # We use a recursive approach with pruning
    # State: (remaining_candidates, current_set, current_size)
    # Upper bound: current_size + len(remaining_candidates)
    # Prune if upper_bound <= best_size

    def branch_and_bound(candidates, current_set, current_size, depth=0):
        """Branch and bound to find maximum valid subset."""
        nonlocal best_set, best_size

        if time.time() > deadline - 3:
            return

        if not candidates:
            return

        # Upper bound: we can add at most len(candidates) more elements
        if current_size + len(candidates) <= best_size:
            return

        # Pick the element with highest conflict count among candidates
        # This helps prune earlier
        x = max(candidates, key=lambda e: conflict_count[e])

        # Option 1: Include x
        new_set = current_set | {x}
        # Filter candidates: remove x and any that conflict with x
        new_candidates = set()
        for y in candidates:
            if y == x:
                continue
            if check_add(y, new_set):
                new_candidates.add(y)

        if current_size + 1 + len(new_candidates) > best_size:
            branch_and_bound(new_candidates, new_set, current_size + 1, depth + 1)

        # Option 2: Exclude x
        new_candidates = candidates - {x}
        if current_size + len(new_candidates) > best_size:
            branch_and_bound(new_candidates, current_set, current_size, depth + 1)

    # Initialize with all elements
    all_elements = set(range(1, max_val + 1))
    # Use a smarter initial bound: start from greedy result
    # and try to improve via B&B on the complement

    # First, let's try B&B starting from the greedy solution
    # We'll try to add elements from the complement
    if time.time() < deadline - 10:
        current_set = set(best_set)
        not_in_set = set(range(1, max_val + 1)) - current_set

        # Sort not_in_set by conflict count (ascending - easier to add first)
        sorted_not_in = sorted(not_in_set, key=lambda x: conflict_count[x])

        # Try adding elements one by one with backtracking
        def bnb_add(candidates, current_set, current_size, depth=0):
            nonlocal best_set, best_size

            if time.time() > deadline - 3:
                return

            if not candidates:
                return

            # Upper bound
            if current_size + len(candidates) <= best_size:
                return

            x = candidates[0]
            rest = candidates[1:]

            # Option 1: Try to add x
            if check_add(x, current_set):
                new_set = current_set | {x}
                if current_size + 1 + len(rest) > best_size:
                    bnb_add(rest, new_set, current_size + 1, depth + 1)

            # Option 2: Skip x
            if current_size + len(rest) > best_size:
                bnb_add(rest, current_set, current_size, depth + 1)

        bnb_add(sorted_not_in, current_set, len(current_set), 0)

    # Strategy 3: Local search with swaps
    for iteration in range(10):
        if time.time() > deadline - 5:
            break

        current_set = set(best_set)
        not_in_set = set(range(1, max_val + 1)) - current_set
        improved = False

        for x in list(current_set):
            if time.time() > deadline - 5:
                break
            current_set.remove(x)
            not_in_set.add(x)

            added = []
            for y in list(not_in_set):
                if y == x:
                    continue
                if check_add(y, current_set):
                    current_set.add(y)
                    not_in_set.remove(y)
                    added.append(y)

            if len(added) > 1:
                improved = True
                if len(current_set) > best_size:
                    best_size = len(current_set)
                    best_set = current_set
                break
            else:
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

    # Strategy 4: More aggressive local search - try removing 2 elements and adding 3+
    if time.time() < deadline - 5:
        current_set = set(best_set)
        not_in_set = set(range(1, max_val + 1)) - current_set

        for _ in range(100):
            if time.time() > deadline - 3:
                break

            if len(current_set) < 2:
                break

            # Pick two elements to remove
            x, y = random.sample(list(current_set), 2)
            current_set.remove(x)
            current_set.remove(y)
            not_in_set.add(x)
            not_in_set.add(y)

            added = []
            for z in list(not_in_set):
                if z in (x, y):
                    continue
                if check_add(z, current_set):
                    current_set.add(z)
                    not_in_set.remove(z)
                    added.append(z)

            if len(added) >= 3 and len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set
            else:
                current_set.add(x)
                current_set.add(y)
                not_in_set.remove(x)
                not_in_set.remove(y)
                for z in added:
                    current_set.remove(z)
                    not_in_set.add(z)

    # Final polish
    current_set = set(best_set)
    not_in_set = set(range(1, max_val + 1)) - current_set
    current_set, not_in_set = try_add_all(current_set, not_in_set)
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = current_set

    return list(best_set)
