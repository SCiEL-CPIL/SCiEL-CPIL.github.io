"""Optimized solver for large integer sets avoiding equal-lcm triples."""
import time
import random
from math import gcd
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill

    N = 128

    # Precompute lcm for all pairs (1..128)
    # lcm(a,b) = a*b//gcd(a,b)
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a, N + 1):
            g = gcd(a, b)
            l = a * b // g
            lcm_table[a][b] = l
            lcm_table[b][a] = l

    # A triple (a,b,c) of distinct elements is "bad" if
    # lcm(a,b) == lcm(a,c) == lcm(b,c)
    # Precompute for each element x, the set of pairs (a,b) with a<b<x
    # such that {a,b,x} is a bad triple.
    # More efficiently: for each pair (a,b), precompute which x > b would form a bad triple.
    # Actually, let's precompute: for each element x, the set of "blocking pairs" among elements < x.
    # blocking_pairs[x] = set of (a,b) with a<b<x such that lcm(a,b)==lcm(a,x)==lcm(b,x)

    # Even better: for each element x, precompute the set of pairs (a,b) from [1,x-1]
    # that would conflict if x is added.
    # For greedy: when adding x, check if any pair in selected conflicts.
    # For local search: when swapping, need to check conflicts.

    # Let's precompute for each x, the list of pairs (a,b) with a<b<x that form bad triple with x.
    # This is O(N^3) = 128^3 ~ 2M, fine.
    blocking_pairs = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for a in range(1, x):
            for b in range(a + 1, x):
                if lcm_table[a][b] == lcm_table[a][x] == lcm_table[b][x]:
                    blocking_pairs[x].add((a, b))

    # For checking if a set S is valid:
    # For each x in S, no pair (a,b) in S with a<b<x should be in blocking_pairs[x].
    # Equivalently, for each pair (a,b) in S, no x in S with x>b such that (a,b) in blocking_pairs[x].

    # Let's also precompute: for each pair (a,b), the set of x > b that conflict.
    conflict_with_pair = {}
    for x in range(1, N + 1):
        for (a, b) in blocking_pairs[x]:
            if (a, b) not in conflict_with_pair:
                conflict_with_pair[(a, b)] = set()
            conflict_with_pair[(a, b)].add(x)

    def is_valid(selected):
        """Check if selected (sorted list) is valid."""
        sel_set = set(selected)
        for i in range(len(selected)):
            x = selected[i]
            for j in range(i):
                a = selected[j]
                # Check all pairs (a, b) where b is between a and x in selected
                # Actually need to check all pairs in selected that are < x
                # More efficient: for each x, check if any pair in selected ∩ [1,x-1] is in blocking_pairs[x]
        # Better approach:
        for x in selected:
            bp = blocking_pairs[x]
            if bp:
                for (a, b) in bp:
                    if a in sel_set and b in sel_set:
                        return False
        return True

    def greedy_with_order(order):
        """Greedy: process elements in given order, add if valid."""
        selected = []
        sel_set = set()
        for x in order:
            # Check if adding x creates a bad triple
            bp = blocking_pairs[x]
            if bp:
                ok = True
                for (a, b) in bp:
                    if a in sel_set and b in sel_set:
                        ok = False
                        break
                if not ok:
                    continue
            selected.append(x)
            sel_set.add(x)
        return selected

    def local_search(selected, max_iterations=1000):
        """Try to improve by removing one and adding another."""
        selected = list(selected)
        sel_set = set(selected)
        improved = True
        iters = 0
        while improved and iters < max_iterations:
            improved = False
            iters += 1
            for i in range(len(selected)):
                removed = selected[i]
                # Try adding each element not in selected
                for x in range(1, N + 1):
                    if x in sel_set:
                        continue
                    # Check if {selected - {removed}} ∪ {x} is valid
                    # Check conflicts for x with remaining elements
                    new_set = sel_set - {removed} | {x}
                    bp = blocking_pairs[x]
                    if bp:
                        conflict = False
                        for (a, b) in bp:
                            if a in new_set and b in new_set:
                                conflict = True
                                break
                        if conflict:
                            continue
                    # Also need to check that removing 'removed' and adding x doesn't create
                    # new conflicts among existing elements... but existing elements were already
                    # valid, and we're only adding x, so we only need to check x's conflicts.
                    # Wait, we also need to ensure that the remaining elements are still valid
                    # without 'removed'. They were valid before, and removing an element can't
                    # create new bad triples, so they remain valid.
                    # So we just need to check x against the remaining set.
                    selected[i] = x
                    sel_set = new_set
                    improved = True
                    break
                if improved:
                    break
        return selected

    best = []
    best_len = 0

    # Run multiple random greedy trials
    trial_count = 0
    while time.time() < deadline:
        # Generate random order
        order = list(range(1, N + 1))
        random.shuffle(order)

        selected = greedy_with_order(order)
        if len(selected) > best_len:
            best = selected
            best_len = len(selected)

        # Local search on this solution
        improved = local_search(selected, max_iterations=200)
        if len(improved) > best_len:
            best = improved
            best_len = len(improved)

        trial_count += 1

        # Periodically check time
        if trial_count % 10 == 0 and time.time() > deadline - 5:
            break

    # Final local search on best
    if time.time() < deadline - 3:
        final = local_search(best, max_iterations=500)
        if len(final) > len(best):
            best = final

    # Ensure nonempty
    if not best:
        best = [1]

    return best
