"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # safety margin before 90s kill

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

    def local_search_improve(selected, reachable, deadline):
        """
        Try to improve the solution via local search (add, 1-for-1 swap, 1-for-2 swap).
        Returns improved selected set and reachable sums.
        """
        selected_set = set(selected)
        reachable = set(reachable)
        best_size = len(selected_set)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            excluded = [x for x in range(1, 101) if x not in selected_set]
            
            # Phase 1: Try adding each excluded element
            for x in excluded:
                if time.time() > deadline:
                    return selected_set, reachable
                if (target - x) not in reachable:
                    selected_set.add(x)
                    new_sums = {s + x for s in reachable if s + x <= target}
                    reachable.update(new_sums)
                    best_size = len(selected_set)
                    improved = True
                    excluded.remove(x)
            
            # Phase 2: Try 1-for-1 swaps (remove a, add x)
            if time.time() < deadline - 1.0:
                for a in list(selected_set):
                    if time.time() > deadline:
                        return selected_set, reachable
                    
                    temp_selected = selected_set - {a}
                    temp_reachable = compute_reachable(temp_selected)
                    
                    for x in excluded:
                        if (target - x) not in temp_reachable:
                            # This swap keeps size same, but may enable future additions
                            # Only accept if it doesn't reduce size (it won't) and potentially helps
                            # Actually, same size swap is only useful if it enables more additions later
                            # For now, skip same-size swaps unless we can immediately add more
                            pass
            
            # Phase 3: Try 1-for-2 swaps (remove a, add x, y) — increases size by 1
            if time.time() < deadline - 2.0:
                for a in list(selected_set):
                    if time.time() > deadline:
                        return selected_set, reachable
                    
                    temp_selected = selected_set - {a}
                    temp_reachable = compute_reachable(temp_selected)
                    
                    # Try adding one element
                    added_one = None
                    for x in excluded:
                        if (target - x) not in temp_reachable:
                            added_one = x
                            break
                    
                    if added_one is not None:
                        # Now try adding a second element
                        temp2_selected = temp_selected | {added_one}
                        temp2_reachable = compute_reachable(temp2_selected)
                        
                        for y in excluded:
                            if y == added_one:
                                continue
                            if (target - y) not in temp2_reachable:
                                # 1-for-2 swap: remove a, add added_one and y
                                candidate_size = len(temp2_selected) + 1
                                if candidate_size > best_size:
                                    selected_set = temp2_selected | {y}
                                    reachable = compute_reachable(selected_set)
                                    best_size = candidate_size
                                    excluded = [x for x in range(1, 101) if x not in selected_set]
                                    improved = True
                                    break
                        
                        if improved:
                            break
                    else:
                        # Try adding two elements without a first add
                        for i, x in enumerate(excluded):
                            if time.time() > deadline:
                                break
                            if (target - x) not in temp_reachable:
                                temp2_reachable = compute_reachable(temp_selected | {x})
                                for y in excluded[i+1:]:
                                    if (target - y) not in temp2_reachable:
                                        candidate_size = len(temp_selected) + 2
                                        if candidate_size > best_size:
                                            selected_set = temp_selected | {x, y}
                                            reachable = compute_reachable(selected_set)
                                            best_size = candidate_size
                                            excluded = [x for x in range(1, 101) if x not in selected_set]
                                            improved = True
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

    # 1. Deterministic Greedy Strategies
    orders = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        list(range(50, 0, -1)) + list(range(100, 50, -1)),
        list(range(100, 0, -2)) + list(range(99, 0, -2)),
        list(range(1, 101, 2)) + list(range(100, 0, -2)),
        list(range(25, 0, -1)) + list(range(75, 0, -1)),
        list(range(100, 0, -1)),  # large first
    ]

    for order in orders:
        if time.time() > deadline:
            break
        sel, reach = greedy_select(order)
        evaluate_and_update(sel, reach)
        if time.time() < deadline - 5.0:
            sel_ls, reach_ls = local_search_improve(sel, reach, deadline)
            evaluate_and_update(sel_ls, reach_ls)

    # 2. Randomized Greedy
    random.seed(42)
    
    while time.time() < deadline - 10.0:
        shuffled = list(universe)
        random.shuffle(shuffled)
        sel, reach = greedy_select(shuffled)
        evaluate_and_update(sel, reach)

    # 3. Final Local Search on the Best Solution Found
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
