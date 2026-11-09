"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import gcd

def construct():
    N = 128
    deadline = time.time() + 50  # safety margin

    # Precompute GCD table for O(1) lookups
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    # Precompute conflict bitmasks: conflict[a] is a bitmask of elements that form a bad triple with a
    # For each pair (a,b), if adding c would create a bad triple, mark c in conflict[a] and conflict[b]
    # Actually, we need: for a given set S and element x, check if any pair (a,b) in S has x in bad_triple[a][b]
    # Better approach: precompute for each x, which pairs (a,b) would conflict with x
    # conflict_with_x[x] = set of pairs (a,b) such that {a,b,x} is a bad triple
    # Then can_add(S, x) checks if any pair in S is in conflict_with_x[x]
    
    # Alternative: for each x, precompute bitmask of elements that "conflict" with x in some sense
    # Actually the cleanest: for each x, precompute a list of (a,b) pairs that form bad triple with x
    # Then can_add(S, x): for each (a,b) in bad_pairs_with[x], check if a in S and b in S
    
    bad_pairs_with = [[] for _ in range(N + 1)]  # bad_pairs_with[x] = list of (a,b) where {a,b,x} is bad
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_table[a][b]
            for c in range(b + 1, N + 1):
                if gcd_table[b][c] == gab and gcd_table[c][a] == gab:
                    bad_pairs_with[c].append((a, b))
                    bad_pairs_with[a].append((b, c))
                    bad_pairs_with[b].append((a, c))

    # For faster checking, use a set for each x
    bad_pairs_set = [set() for _ in range(N + 1)]
    for x in range(1, N + 1):
        for pair in bad_pairs_with[x]:
            bad_pairs_set[x].add(pair)

    # Even better: use bitmask representation
    # selected_mask: integer where bit i is set if i is in selected set
    # For can_add(selected_mask, x): check if any pair (a,b) in bad_pairs_with[x] has both a and b in selected_mask
    
    def can_add_mask(selected_mask, x):
        """Check if x can be added to selected (represented as bitmask) without creating a bad triple."""
        for (a, b) in bad_pairs_with[x]:
            if (selected_mask >> a) & 1 and (selected_mask >> b) & 1:
                return False
        return True

    def greedy_build(order):
        """Greedy build following given order using bitmask."""
        selected_mask = 0
        for x in order:
            if can_add_mask(selected_mask, x):
                selected_mask |= (1 << x)
        return selected_mask

    def mask_to_list(mask):
        """Convert bitmask to list of elements."""
        return [i for i in range(1, N + 1) if (mask >> i) & 1]

    def local_search(selected_mask, time_limit=5.0):
        """Local search using bitmask representation."""
        selected_mask = selected_mask
        start = time.time()
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            # Try adding any element not in selected
            for x in range(1, N + 1):
                if not ((selected_mask >> x) & 1):
                    if can_add_mask(selected_mask, x):
                        selected_mask |= (1 << x)
                        improved = True
                        break
            # If we can't add, try 1-swap: remove one, add another
            if not improved:
                for x in range(1, N + 1):
                    if not ((selected_mask >> x) & 1):
                        continue
                    remaining_mask = selected_mask & ~(1 << x)
                    for y in range(1, N + 1):
                        if (selected_mask >> y) & 1:
                            continue
                        if can_add_mask(remaining_mask, y):
                            selected_mask = remaining_mask | (1 << y)
                            improved = True
                            break
                    if improved:
                        break
            # If still no improvement, try 2-swap: remove two, add two
            if not improved and time.time() - start < time_limit - 0.5:
                sel_list = mask_to_list(selected_mask)
                for i in range(len(sel_list)):
                    if time.time() - start >= time_limit - 0.5:
                        break
                    for j in range(i + 1, len(sel_list)):
                        if time.time() - start >= time_limit - 0.5:
                            break
                        x1, x2 = sel_list[i], sel_list[j]
                        remaining_mask = selected_mask & ~(1 << x1) & ~(1 << x2)
                        found = False
                        for y1 in range(1, N + 1):
                            if found or ((selected_mask >> y1) & 1):
                                continue
                            if not can_add_mask(remaining_mask, y1):
                                continue
                            for y2 in range(y1 + 1, N + 1):
                                if found or ((selected_mask >> y2) & 1):
                                    continue
                                if can_add_mask(remaining_mask | (1 << y1), y2):
                                    selected_mask = remaining_mask | (1 << y1) | (1 << y2)
                                    improved = True
                                    found = True
                                    break
                        if improved:
                            break
                    if improved:
                        break
        return selected_mask

    best_mask = 0
    best_size = 0

    # Multiple restarts with different orderings
    base_order = list(range(1, N + 1))

    # Try natural order first
    result_mask = greedy_build(base_order)
    result_mask = local_search(result_mask, time_limit=3.0)
    result_size = bin(result_mask).count('1')
    if result_size > best_size:
        best_mask = result_mask
        best_size = result_size

    # Try conflict-ordered (fewest conflicts first)
    conflict_ordered = sorted(range(1, N + 1), key=lambda x: len(bad_pairs_with[x]))
    result_mask = greedy_build(conflict_ordered)
    result_mask = local_search(result_mask, time_limit=3.0)
    result_size = bin(result_mask).count('1')
    if result_size > best_size:
        best_mask = result_mask
        best_size = result_size

    # Random restarts
    restart_count = 0
    no_improve_count = 0
    while time.time() < deadline:
        if random.random() < 0.3:
            # Perturb: start from best and randomly remove some, then rebuild
            best_list = mask_to_list(best_mask)
            start_set = set(best_list)
            for _ in range(random.randint(1, 5)):
                if start_set:
                    start_set.remove(random.choice(list(start_set)))
            remaining_order = [x for x in range(1, N + 1) if x not in start_set]
            random.shuffle(remaining_order)
            order = list(start_set) + remaining_order
            result_mask = greedy_build(order)
        else:
            random.shuffle(base_order)
            result_mask = greedy_build(base_order)
        result_mask = local_search(result_mask, time_limit=2.0)
        result_size = bin(result_mask).count('1')
        if result_size > best_size:
            best_mask = result_mask
            best_size = result_size
            no_improve_count = 0
        else:
            no_improve_count += 1
        restart_count += 1
        if no_improve_count > 50 and best_size >= 80:
            break

    # Final polish: try to add any remaining elements
    for x in range(1, N + 1):
        if not ((best_mask >> x) & 1) and can_add_mask(best_mask, x):
            best_mask |= (1 << x)

    best_list = mask_to_list(best_mask)
    return best_list if best_list else [1]
# EVOLVE-BLOCK-END
