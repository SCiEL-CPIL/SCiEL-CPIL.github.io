"""Rewritten program for large integer sets with no two-other-product divisibility."""
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # Candidate set: all integers from 2 to 256
    all_candidates = list(range(2, 257))
    n_total = len(all_candidates)

    # Precompute conflict hyperedges: for each triple (a,b,c) distinct,
    # check if a | b*c or b | a*c or c | a*b
    # We model this as: vertex v conflicts with pair (x,y) if v | x*y
    # So the "forbidden" condition is: no vertex a in S such that a | b*c for distinct b,c in S

    # Build for each vertex a, the set of pairs (b,c) with b!=c, b,c != a, such that a | b*c
    # Actually, let's think differently:
    # A set S is valid iff for all distinct a,b,c in S: a does NOT divide b*c
    # Equivalently: for each a in S, and for each pair {b,c} subset of S\{a} with b!=c, we need a ∤ b*c

    # Precompute: for each a, which pairs (b,c) cause conflict (i.e., a | b*c)
    # conflict_pairs[a] = set of frozenset({b,c}) where b!=c, b,c != a, and a | b*c

    conflict_pairs = {}
    for a in all_candidates:
        pairs = set()
        for i in range(n_total):
            b = all_candidates[i]
            if b == a:
                continue
            for j in range(i + 1, n_total):
                c = all_candidates[j]
                if c == a or c == b:
                    continue
                if (b * c) % a == 0:
                    pairs.add((min(b, c), max(b, c)))
        conflict_pairs[a] = pairs

    # For branch and bound:
    # State: current_set (vertices chosen), available (vertices not yet decided)
    # A vertex v can be added to current_set iff for all pairs (x,y) in current_set with x!=y:
    #   v does NOT divide x*y, AND x does NOT divide v*y, AND y does NOT divide v*x
    # Which is equivalent to: no triple (v, x, y) in current_set ∪ {v} has the divisibility property

    def can_add_vertex(v, current_set):
        """Check if v can be added to current_set."""
        cur_list = list(current_set)
        n = len(cur_list)
        for i in range(n):
            x = cur_list[i]
            for j in range(i + 1, n):
                y = cur_list[j]
                # Check if v | x*y
                if (x * y) % v == 0:
                    return False
                # Check if x | v*y
                if (v * y) % x == 0:
                    return False
                # Check if y | v*x
                if (v * x) % y == 0:
                    return False
        return True

    # Upper bound: for remaining vertices, estimate max additional vertices we can add
    # Simple bound: len(available)
    # Better bound: count how many available vertices can be added to current_set
    def upper_bound(current_set, available):
        """Upper bound on additional vertices that can be added."""
        count = 0
        for v in available:
            if can_add_vertex(v, current_set):
                count += 1
        return count

    # Branch and bound
    best_set = set()
    best_size = 0

    def branch_and_bound(current_set, available, depth=0):
        nonlocal best_set, best_size

        if time.time() > deadline:
            return

        # Update best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)

        if not available:
            return

        # Upper bound pruning
        ub = upper_bound(current_set, available)
        if len(current_set) + ub <= best_size:
            return

        # Choose next vertex to branch on
        # Heuristic: pick vertex with highest "constraint" (most conflicts)
        # Or just pick first available
        v = available[0]
        remaining = available[1:]

        # Branch 1: include v
        if can_add_vertex(v, current_set):
            new_current = current_set | {v}
            # Filter remaining: remove vertices that conflict with v given current_set
            # A vertex u in remaining can only be added if it's compatible with v and current_set
            filtered_remaining = []
            for u in remaining:
                if can_add_vertex(u, new_current):
                    filtered_remaining.append(u)
            branch_and_bound(new_current, filtered_remaining, depth + 1)

        # Branch 2: exclude v
        branch_and_bound(current_set, remaining, depth + 1)

    # Initial greedy solution for a good lower bound
    # Try multiple orderings
    random.seed(42)
    orderings = [
        sorted(all_candidates, reverse=True),
        sorted(all_candidates),
    ]
    for _ in range(10):
        shuffled = all_candidates[:]
        random.shuffle(shuffled)
        orderings.append(shuffled)

    for ordering in orderings:
        if time.time() > deadline - 10:
            break
        current = set()
        for c in ordering:
            if can_add_vertex(c, current):
                current.add(c)
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)

    # Run branch and bound with time limit
    # Use a smarter ordering: sort by number of conflicts (descending) to prune earlier
    conflict_counts = {v: len(conflict_pairs[v]) for v in all_candidates}
    sorted_available = sorted(all_candidates, key=lambda x: -conflict_counts[x])

    if time.time() < deadline - 2:
        branch_and_bound(set(), sorted_available)

    # Final verification
    result = sorted(best_set)

    # Verify validity
    def is_valid(s):
        s_list = list(s)
        n = len(s_list)
        for i in range(n):
            a = s_list[i]
            for j in range(i + 1, n):
                b = s_list[j]
                for k in range(j + 1, n):
                    c = s_list[k]
                    if (b * c) % a == 0:
                        return False
                    if (a * c) % b == 0:
                        return False
                    if (a * b) % c == 0:
                        return False
        return True

    if not is_valid(result):
        # Fallback
        current = set()
        for c in all_candidates:
            if can_add_vertex(c, current):
                current.add(c)
        result = sorted(current)

    return [int(x) for x in result]

# EVOLVE-BLOCK-END
