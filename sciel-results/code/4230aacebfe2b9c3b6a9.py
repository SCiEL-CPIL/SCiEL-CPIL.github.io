"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # safety margin before 90s kill

    target = 200
    max_val = 100
    universe = list(range(1, max_val + 1))

    def compute_reachable(selected):
        """Compute reachable sums up to target for a given set."""
        reachable = {0}
        for x in selected:
            new_sums = {s + x for s in reachable if s + x <= target}
            reachable.update(new_sums)
        return reachable

    def is_valid(selected):
        """Check if no subset of selected sums to target."""
        reachable = {0}
        for x in selected:
            new_sums = {s + x for s in reachable if s + x <= target}
            if target in new_sums:
                return False
            reachable.update(new_sums)
        return True

    def greedy_select(order):
        """
        Greedily select elements in the given order.
        Returns the selected set and the reachable sums.
        """
        selected = set()
        reachable = {0}

        for x in order:
            if (target - x) in reachable:
                continue
            selected.add(x)
            new_sums = {s + x for s in reachable if s + x <= target}
            reachable.update(new_sums)

        return selected, reachable

    def min_degree_greedy(universe_set, deadline):
        """
        Hypergraph-inspired greedy: at each step, pick the element x
        that has the minimum 'degree' — i.e., the fewest subsets of
        remaining elements that could combine with x to sum to target.
        This preserves maximum flexibility for future selections.
        """
        selected = set()
        remaining = set(universe_set)
        reachable = {0}

        while remaining and time.time() < deadline:
            best_x = None
            best_degree = float('inf')

            for x in remaining:
                # Check if x is safe to add
                if (target - x) in reachable:
                    continue

                # Compute degree: number of subsets of (remaining - {x})
                # that sum to (target - x)
                # This tells us how many "forbidden combinations" x participates in
                # with the remaining elements
                rem_without_x = remaining - {x}
                # Compute what sums are reachable from rem_without_x
                rem_reachable = {0}
                for y in rem_without_x:
                    new_sums = {s + y for s in rem_reachable if s + y <= target}
                    rem_reachable.update(new_sums)

                degree = 1 if (target - x) in rem_reachable else 0

                if degree < best_degree:
                    best_degree = degree
                    best_x = x

                # Early exit: if we found a zero-degree element, it's optimal
                if best_degree == 0:
                    break

            if best_x is None:
                break

            selected.add(best_x)
            remaining.remove(best_x)
            new_sums = {s + best_x for s in reachable if s + best_x <= target}
            reachable.update(new_sums)

        return selected, reachable

    def local_search_improve(selected, reachable, deadline):
        """
        Try to improve via:
        1. Adding any element that doesn't create target sum
        2. 1-for-2 swaps: remove a, add x and y (net +1)
        3. 2-for-3 swaps: remove a,b, add x,y,z (net +1)
        """
        selected_set = set(selected)
        reachable = set(reachable)
        best_size = len(selected_set)
        improved = True

        while improved and time.time() < deadline:
            improved = False
            excluded = [x for x in range(1, 101) if x not in selected_set]

            # Phase 1: Try adding each excluded element
            for x in excluded[:]:
                if time.time() > deadline:
                    return selected_set, reachable
                if (target - x) not in reachable:
                    selected_set.add(x)
                    new_sums = {s + x for s in reachable if s + x <= target}
                    reachable.update(new_sums)
                    best_size = len(selected_set)
                    improved = True
                    excluded.remove(x)

            # Phase 2: 1-for-2 swaps (remove a, add x and y) — increases size by 1
            if time.time() < deadline - 1.0:
                for a in list(selected_set):
                    if time.time() > deadline:
                        return selected_set, reachable

                    temp_selected = selected_set - {a}
                    temp_reachable = compute_reachable(temp_selected)

                    # Find all elements we can add after removing a
                    addable = []
                    for x in excluded:
                        if (target - x) not in temp_reachable:
                            addable.append(x)

                    if len(addable) >= 2:
                        # Try pairs from addable
                        for i in range(len(addable)):
                            x = addable[i]
                            temp2 = temp_selected | {x}
                            temp2_reach = compute_reachable(temp2)
                            for j in range(i + 1, len(addable)):
                                y = addable[j]
                                if (target - y) not in temp2_reach:
                                    new_sel = temp2 | {y}
                                    if len(new_sel) > best_size:
                                        selected_set = new_sel
                                        reachable = compute_reachable(selected_set)
                                        best_size = len(selected_set)
                                        excluded = [x for x in range(1, 101) if x not in selected_set]
                                        improved = True
                                        break
                            if improved:
                                break
                    elif len(addable) == 1:
                        # Can't do 1-for-2, but maybe 1-for-1 helps future
                        pass

                    if improved:
                        break

            # Phase 3: 2-for-3 swaps (remove a,b, add x,y,z) — increases size by 1
            if time.time() < deadline - 2.0 and len(selected_set) >= 2:
                sel_list = list(selected_set)
                for i in range(len(sel_list)):
                    if time.time() > deadline:
                        return selected_set, reachable
                    a = sel_list[i]
                    for j in range(i + 1, len(sel_list)):
                        b = sel_list[j]
                        temp_selected = selected_set - {a, b}
                        temp_reachable = compute_reachable(temp_selected)

                        addable = []
                        for x in excluded:
                            if (target - x) not in temp_reachable:
                                addable.append(x)

                        if len(addable) >= 3:
                            for x in addable:
                                t2 = temp_selected | {x}
                                t2r = compute_reachable(t2)
                                for y in addable:
                                    if y == x:
                                        continue
                                    if (target - y) not in t2r:
                                        t3 = t2 | {y}
                                        t3r = compute_reachable(t3)
                                        for z in addable:
                                            if z in (x, y):
                                                continue
                                            if (target - z) not in t3r:
                                                new_sel = t3 | {z}
                                                if len(new_sel) > best_size:
                                                    selected_set = new_sel
                                                    reachable = compute_reachable(selected_set)
                                                    best_size = len(selected_set)
                                                    excluded = [x for x in range(1, 101) if x not in selected_set]
                                                    improved = True
                                                    break
                                        if improved:
                                            break
                                    if improved:
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                    if improved:
                        break

        return selected_set, reachable

    best_solution = None
    best_size = 0

    def evaluate_and_update(candidates, reachable):
        nonlocal best_solution, best_size
        if not candidates:
            return
        size = len(candidates)
        if size > best_size:
            best_solution = set(candidates)
            best_size = size

    # Strategy 1: Min-degree greedy (hypergraph-inspired)
    if time.time() < deadline - 5.0:
        sel, reach = min_degree_greedy(set(universe), deadline)
        evaluate_and_update(sel, reach)
        if time.time() < deadline - 3.0:
            sel_ls, reach_ls = local_search_improve(sel, reach, deadline)
            evaluate_and_update(sel_ls, reach_ls)

    # Strategy 2: Standard greedy with various orders
    orders = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        list(range(50, 0, -1)) + list(range(100, 50, -1)),
        list(range(100, 0, -2)) + list(range(99, 0, -2)),
        list(range(1, 101, 2)) + list(range(100, 0, -2)),
        list(range(25, 0, -1)) + list(range(75, 0, -1)),
    ]

    for order in orders:
        if time.time() > deadline - 5.0:
            break
        sel, reach = greedy_select(order)
        evaluate_and_update(sel, reach)
        if time.time() < deadline - 3.0:
            sel_ls, reach_ls = local_search_improve(sel, reach, deadline)
            evaluate_and_update(sel_ls, reach_ls)

    # Strategy 3: Randomized greedy with min-degree bias
    random.seed(42)
    while time.time() < deadline - 5.0:
        # Create a biased random order: larger numbers get slight priority
        # but with randomness to explore different structures
        biased_order = sorted(universe, key=lambda x: -x + random.gauss(0, 10))
        sel, reach = greedy_select(biased_order)
        evaluate_and_update(sel, reach)

        # Also try pure random
        shuffled = list(universe)
        random.shuffle(shuffled)
        sel, reach = greedy_select(shuffled)
        evaluate_and_update(sel, reach)

    # Strategy 4: Min-degree greedy with random tie-breaking
    if time.time() < deadline - 5.0:
        for _ in range(5):
            if time.time() > deadline - 3.0:
                break
            # Perturb the universe slightly by randomizing tie-breaking
            sel, reach = min_degree_greedy(set(universe), deadline)
            evaluate_and_update(sel, reach)
            if time.time() < deadline - 2.0:
                sel_ls, reach_ls = local_search_improve(sel, reach, deadline)
                evaluate_and_update(sel_ls, reach_ls)

    # Final local search on the best solution found
    if best_solution is not None:
        best_reachable = compute_reachable(best_solution)
        best_solution, best_reachable = local_search_improve(best_solution, best_reachable, deadline)

    # Fallback
    if best_solution is None:
        best_solution, _ = greedy_select(list(range(100, 0, -1)))

    # Final Verification
    reachable = {0}
    valid = True
    for x in best_solution:
        new_sums = {s + x for s in reachable if s + x <= target}
        if target in new_sums:
            valid = False
            break
        reachable.update(new_sums)

    if not valid:
        fallback_sel, _ = greedy_select(list(range(100, 0, -1)))
        return list(fallback_sel)

    return list(best_solution)

# EVOLVE-BLOCK-END
