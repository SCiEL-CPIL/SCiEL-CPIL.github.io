"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import gcd
from itertools import combinations

def construct():
    N = 128
    deadline = time.time() + 50  # safety margin

    # Precompute GCD table for O(1) lookups
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    # Precompute all bad triples: bad_triple[a][b] = set of c such that {a,b,c} is bad
    # Use a 2D list of sets for O(1) lookup
    bad_triple = [[set() for _ in range(N + 1)] for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_table[a][b]
            for c in range(b + 1, N + 1):
                if gcd_table[b][c] == gab and gcd_table[c][a] == gab:
                    bad_triple[a][b].add(c)
                    bad_triple[b][a].add(c)
                    bad_triple[a][c].add(b)
                    bad_triple[c][a].add(b)
                    bad_triple[b][c].add(a)
                    bad_triple[c][b].add(a)

    # Count conflicts for each number (how many bad triples it participates in)
    conflict_count = [0] * (N + 1)
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            cnt = len(bad_triple[a][b])
            conflict_count[a] += cnt
            conflict_count[b] += cnt

    def can_add(selected_set, x):
        """Check if x can be added to selected_set without creating a bad triple."""
        sel = list(selected_set)
        for i in range(len(sel)):
            for j in range(i + 1, len(sel)):
                a, b = sel[i], sel[j]
                if x in bad_triple[a][b]:
                    return False
        return True

    def greedy_build(order):
        """Greedy build following given order."""
        selected = set()
        for x in order:
            if can_add(selected, x):
                selected.add(x)
        return selected

    def local_search(selected_set, time_limit=5.0):
        """Local search: try to add elements, swap elements, 2-swaps."""
        selected = set(selected_set)
        start = time.time()
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            # Try adding any element not in selected
            for x in range(1, N + 1):
                if x not in selected:
                    if can_add(selected, x):
                        selected.add(x)
                        improved = True
                        break
            # If we can't add, try 1-swap: remove one, add another
            if not improved:
                for x in list(selected):
                    remaining = selected - {x}
                    for y in range(1, N + 1):
                        if y not in selected:
                            if can_add(remaining, y):
                                selected = remaining | {y}
                                improved = True
                                break
                    if improved:
                        break
            # If still no improvement, try 2-swap: remove two, add two
            if not improved and time.time() - start < time_limit - 0.5:
                sel_list = list(selected)
                for i in range(len(sel_list)):
                    if time.time() - start >= time_limit - 0.5:
                        break
                    for j in range(i + 1, len(sel_list)):
                        if time.time() - start >= time_limit - 0.5:
                            break
                        x1, x2 = sel_list[i], sel_list[j]
                        remaining = selected - {x1, x2}
                        found = False
                        for y1 in range(1, N + 1):
                            if y1 in selected or found:
                                continue
                            if not can_add(remaining, y1):
                                continue
                            for y2 in range(y1 + 1, N + 1):
                                if y2 in selected or found:
                                    continue
                                if can_add(remaining | {y1}, y2):
                                    selected = remaining | {y1, y2}
                                    improved = True
                                    found = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
        return selected

    best = set()

    # Multiple restarts with different orderings
    base_order = list(range(1, N + 1))

    # Try natural order first
    result = greedy_build(base_order)
    result = local_search(result, time_limit=3.0)
    if len(result) > len(best):
        best = set(result)

    # Try conflict-ordered (fewest conflicts first)
    conflict_ordered = sorted(range(1, N + 1), key=lambda x: conflict_count[x])
    result = greedy_build(conflict_ordered)
    result = local_search(result, time_limit=3.0)
    if len(result) > len(best):
        best = set(result)

    # Random restarts
    restart_count = 0
    no_improve_count = 0
    while time.time() < deadline:
        if random.random() < 0.3:
            # Perturb: start from best and randomly remove some, then rebuild
            start_set = set(best)
            for _ in range(random.randint(1, 5)):
                if start_set:
                    start_set.remove(random.choice(list(start_set)))
            remaining_order = [x for x in range(1, N + 1) if x not in start_set]
            random.shuffle(remaining_order)
            order = list(start_set) + remaining_order
            result = greedy_build(order)
        else:
            random.shuffle(base_order)
            result = greedy_build(base_order)
        result = local_search(result, time_limit=2.0)
        if len(result) > len(best):
            best = set(result)
            no_improve_count = 0
        else:
            no_improve_count += 1
        restart_count += 1
        if no_improve_count > 50 and len(best) >= 80:
            break

    # Final polish: try to add any remaining elements
    for x in range(1, N + 1):
        if x not in best and can_add(best, x):
            best.add(x)

    return list(best) if best else [1]
# EVOLVE-BLOCK-END
