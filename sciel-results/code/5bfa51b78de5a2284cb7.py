"""Structural decomposition: separate the easy part (odds>128) from the hard part (evens, odds<=128)."""
import time
import random
from math import gcd
from fractions import Fraction

def construct():
    start_time = time.time()
    DEADLINE = 80.0

    def elapsed():
        return time.time() - start_time

    N = 256
    numbers = list(range(2, N + 1))
    n = len(numbers)
    num_to_idx = {v: i for i, v in enumerate(numbers)}

    # Precompute conflicts
    conflict = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = numbers[i], numbers[j]
            g = gcd(a, b)
            l = a * b // g
            if l <= N:
                conflict[i].add(j)
                conflict[j].add(i)

    weights = [1.0 / numbers[i] for i in range(n)]

    # Structural decomposition:
    # Group A: odds > 128 (always mutually compatible, compatible with all evens)
    # Group B: everything else (evens 2..256, odds 3..127)
    # Within Group B, we need to find max weight independent set.
    # Group A elements can always be added.

    group_a = [i for i in range(n) if numbers[i] % 2 == 1 and numbers[i] > 128]
    group_b = [i for i in range(n) if i not in set(group_a)]

    # Conflicts within group_b only
    b_conflict = [set() for _ in range(len(group_b))]
    b_idx_map = {i: k for k, i in enumerate(group_b)}
    for k, i in enumerate(group_b):
        for j in conflict[i]:
            if j in b_idx_map:
                b_conflict[k].add(b_idx_map[j])

    b_weights = [weights[i] for i in group_b]
    m = len(group_b)

    # Branch and bound for max weight independent set on group_b
    # Order vertices by weight descending for better bounds
    order = sorted(range(m), key=lambda k: -b_weights[k])
    order_inv = [0] * m
    for pos, k in enumerate(order):
        order_inv[k] = pos

    # Reorder conflicts
    b_conflict_ordered = [set() for _ in range(m)]
    for k in range(m):
        for j in b_conflict[k]:
            b_conflict_ordered[k].add(j)

    best_sum = 0.0
    best_set = set()

    # Upper bound: sum of all remaining weights (loose but fast)
    # Better: for each vertex, it conflicts with some others, so we can use
    # a greedy coloring bound

    def greedy_coloring_bound(conflict, remaining):
        """Upper bound via greedy coloring of the subgraph on remaining vertices."""
        if not remaining:
            return 0.0
        # Simple bound: sum of weights of remaining
        # Better: partition into independent sets using greedy coloring
        # For now use sum of weights as bound (can be improved)
        total = 0.0
        for k in remaining:
            total += b_weights[k]
        return total

    # Branch and bound
    selected = set()
    available = set(range(m))

    def branch_bound(selected, available, current_sum, depth=0):
        nonlocal best_sum, best_set

        if elapsed() > DEADLINE:
            return

        # Upper bound
        ub = current_sum + sum(b_weights[k] for k in available)
        if ub <= best_sum + 1e-12:
            return

        if not available:
            if current_sum > best_sum:
                best_sum = current_sum
                best_set = set(selected)
            return

        # Choose next vertex: highest weight in available
        # Find vertex with highest weight
        v = max(available, key=lambda k: b_weights[k])

        # Branch 1: include v
        new_selected = selected | {v}
        new_available = set()
        for k in available:
            if k != v and k not in b_conflict_ordered[v]:
                new_available.add(k)
        branch_bound(new_selected, new_available, current_sum + b_weights[v], depth + 1)

        if elapsed() > DEADLINE:
            return

        # Branch 2: exclude v
        new_available = available - {v}
        branch_bound(selected, new_available, current_sum, depth + 1)

    # This is exponential, so we need a better approach for m~180 vertices.
    # Instead, use a different strategy:
    # 1. Start with greedy solution
    # 2. Use local search with 1- and 2-swap moves
    # 3. Use simulated annealing with better move operators

    # Greedy: process in weight order, add if no conflict with selected
    greedy_selected = set()
    greedy_sum = 0.0
    for k in order:
        if k not in greedy_selected:
            # Check if k conflicts with any selected
            can_add = True
            for s in greedy_selected:
                if s in b_conflict_ordered[k]:
                    can_add = False
                    break
            if can_add:
                greedy_selected.add(k)
                greedy_sum += b_weights[k]

    best_sum = greedy_sum
    best_set = set(greedy_selected)

    # Local search with 1-swap (remove one, add one) and 2-swap (remove two, add two)
    def local_search(selected, current_sum, max_iters=500):
        nonlocal best_sum, best_set

        improved = True
        iters = 0
        while improved and iters < max_iters and elapsed() < DEADLINE - 2:
            improved = False
            iters += 1

            # Try 1-removal + greedy fill
            for s in list(selected):
                if elapsed() > DEADLINE - 2:
                    break
                new_sel = selected - {s}
                # Greedily add back what we can
                for k in order:
                    if k not in new_sel:
                        can_add = True
                        for t in new_sel:
                            if t in b_conflict_ordered[k]:
                                can_add = False
                                break
                        if can_add:
                            new_sel.add(k)
                new_sum = sum(b_weights[k] for k in new_sel)
                if new_sum > current_sum + 1e-12:
                    selected = new_sel
                    current_sum = new_sum
                    improved = True
                    break

            if improved or elapsed() > DEADLINE - 2:
                continue

            # Try 1-swap: remove one, add one
            non_selected = set(range(m)) - selected
            for s in list(selected):
                if elapsed() > DEADLINE - 2:
                    break
                new_sel = selected - {s}
                for k in non_selected:
                    if elapsed() > DEADLINE - 2:
                        break
                    can_add = True
                    for t in new_sel:
                        if t in b_conflict_ordered[k]:
                            can_add = False
                            break
                    if can_add:
                        new_sum = current_sum - b_weights[s] + b_weights[k]
                        if new_sum > current_sum + 1e-12:
                            selected = new_sel | {k}
                            current_sum = new_sum
                            improved = True
                            non_selected = set(range(m)) - selected
                            break
                if improved:
                    break

            if improved or elapsed() > DEADLINE - 2:
                continue

            # Try 2-removal + greedy fill
            sel_list = list(selected)
            for i in range(len(sel_list)):
                if elapsed() > DEADLINE - 2:
                    break
                for j in range(i + 1, len(sel_list)):
                    if elapsed() > DEADLINE - 2:
                        break
                    s1, s2 = sel_list[i], sel_list[j]
                    new_sel = selected - {s1, s2}
                    for k in order:
                        if k not in new_sel:
                            can_add = True
                            for t in new_sel:
                                if t in b_conflict_ordered[k]:
                                    can_add = False
                                    break
                            if can_add:
                                new_sel.add(k)
                    new_sum = sum(b_weights[k] for k in new_sel)
                    if new_sum > current_sum + 1e-12:
                        selected = new_sel
                        current_sum = new_sum
                        improved = True
                        break
                if improved:
                    break

        if current_sum > best_sum:
            best_sum = current_sum
            best_set = set(selected)
        return selected, current_sum

    # Run local search from greedy
    sel, s = local_search(greedy_selected, greedy_sum, max_iters=200)
    best_sum = s
    best_set = set(sel)

    # Try different greedy orders
    random.seed(42)
    for trial in range(5):
        if elapsed() > DEADLINE - 10:
            break
        # Random perturbation of order
        perturbed_order = list(order)
        random.shuffle(perturbed_order)
        # Keep top 20% in weight order, shuffle rest
        top_k = max(1, len(perturbed_order) // 5)
        top = sorted(perturbed_order[:top_k], key=lambda k: -b_weights[k])
        rest = perturbed_order[top_k:]
        random.shuffle(rest)
        perturbed_order = top + rest

        greedy_selected = set()
        greedy_sum = 0.0
        for k in perturbed_order:
            can_add = True
            for t in greedy_selected:
                if t in b_conflict_ordered[k]:
                    can_add = False
                    break
            if can_add:
                greedy_selected.add(k)
                greedy_sum += b_weights[k]

        sel, s = local_search(greedy_selected, greedy_sum, max_iters=100)
        if s > best_sum:
            best_sum = s
            best_set = set(sel)

    # Simulated annealing
    if elapsed() < DEADLINE - 15:
        sa_temp = 0.02
        sa_cool = 0.997
        sa_mask = set(best_set)
        sa_sum = best_sum
        best_sa_sum = sa_sum
        best_sa_set = set(sa_mask)

        non_sel = set(range(m)) - sa_mask
        for it in range(3000):
            if elapsed() > DEADLINE - 5:
                break
            if it % 50 == 0:
                sa_temp *= sa_cool

            if not sa_mask or not non_sel:
                break

            s = random.choice(list(sa_mask))
            k = random.choice(list(non_sel))

            new_mask = sa_mask - {s} | {k}
            # Check validity
            valid = True
            for t in new_mask:
                if t != k and t in b_conflict_ordered[k]:
                    valid = False
                    break

            if not valid:
                continue

            new_sum = sa_sum - b_weights[s] + b_weights[k]
            delta = new_sum - sa_sum

            if delta > 0 or random.random() < pow(2.71828, delta / max(sa_temp, 1e-10)):
                sa_mask = new_mask
                sa_sum = new_sum
                non_sel = set(range(m)) - sa_mask
                if sa_sum > best_sa_sum:
                    best_sa_sum = sa_sum
                    best_sa_set = set(sa_mask)

        if best_sa_sum > best_sum:
            best_sum = best_sa_sum
            best_set = set(best_sa_set)
            # Polish
            if elapsed() < DEADLINE - 3:
                sel, s = local_search(best_sa_set, best_sa_sum, max_iters=50)
                if s > best_sum:
                    best_sum = s
                    best_set = set(sel)

    # Build final result: group_a + best_set from group_b
    result_indices = set(group_a) | set(best_set)
    # Convert back to original indices
    b_set_to_orig = {k: group_b[k] for k in best_set}
    result_indices = set(group_a) | set(b_set_to_orig.values())

    result = [numbers[i] for i in result_indices]
    result.sort()

    if not result:
        result = [2]

    return result
