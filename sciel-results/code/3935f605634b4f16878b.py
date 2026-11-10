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

    def greedy_select(order):
        """
        Greedily select elements in the given order.
        Returns the selected set and the reachable sums.
        """
        selected = set()
        reachable = {0}
        
        for x in order:
            # Check if adding x creates a sum of 200
            # We need to check if there exists s in reachable such that s + x == 200
            # i.e., 200 - x is in reachable
            if (target - x) in reachable:
                continue
            
            # If safe, add x
            selected.add(x)
            # Update reachable sums
            # Only sums <= 200 matter for future checks
            new_sums = {s + x for s in reachable if s + x <= target}
            reachable.update(new_sums)
            
        return selected, reachable

    def local_search_improve(selected, reachable, deadline):
        """
        Try to improve the solution via local search (add/swap operations).
        Returns improved selected set and reachable sums.
        """
        selected_set = set(selected)
        best_size = len(selected_set)
        improved = True
        
        while improved and time.time() < deadline:
            improved = False
            excluded = [x for x in range(1, 101) if x not in selected_set]
            
            # Try adding each excluded element
            for x in excluded:
                if time.time() > deadline:
                    break
                if (target - x) not in reachable:
                    selected_set.add(x)
                    new_sums = {s + x for s in reachable if s + x <= target}
                    reachable.update(new_sums)
                    best_size = len(selected_set)
                    improved = True
            
            # Try swaps: remove one, add one or two
            # We iterate over a copy of the selected set to avoid modification during iteration issues
            # Note: This part is expensive, so we limit it or break early if time is tight
            if time.time() < deadline - 2.0:
                for a in list(selected_set):
                    if time.time() > deadline:
                        break
                    
                    # Temporarily remove a
                    temp_selected = selected_set - {a}
                    
                    # Recompute reachable without a
                    temp_reachable = {0}
                    for s in temp_selected:
                        new_sums = {v + s for v in temp_reachable if v + s <= target}
                        temp_reachable.update(new_sums)
                    
                    # Try adding one excluded element
                    for x in excluded:
                        if x in temp_selected:
                            continue
                        if (target - x) not in temp_reachable:
                            candidate_size = len(temp_selected) + 1
                            if candidate_size > best_size:
                                selected_set = temp_selected | {x}
                                # Update reachable for the new set
                                new_reachable = temp_reachable
                                new_sums = {v + x for v in new_reachable if v + x <= target}
                                new_reachable.update(new_sums)
                                reachable = new_reachable
                                best_size = candidate_size
                                excluded = [x for x in range(1, 101) if x not in selected_set]
                                improved = True
                                break # Restart search with new state
                    
                    # Try adding two excluded elements (only if we didn't already improve with 1)
                    if not improved and time.time() < deadline - 1.0:
                        # This is O(N^2) which might be too slow, so we sample or limit
                        # For now, let's just stick to single swaps in the main loop to keep it fast
                        # Or do a limited check
                        pass

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
        list(range(100, 0, -1)),  # Descending
        list(range(1, 101)),       # Ascending
        list(range(50, 0, -1)) + list(range(100, 50, -1)),  # Mid then High
        list(range(100, 0, -2)) + list(range(99, 0, -2)),    # Evens then Odds
        list(range(1, 101, 2)) + list(range(100, 0, -2)),    # Odds then Evens
    ]

    for order in orders:
        if time.time() > deadline:
            break
        sel, reach = greedy_select(order)
        evaluate_and_update(sel, reach)
        # Apply local search to this specific greedy result
        if time.time() < deadline - 5.0:
            sel_ls, reach_ls = local_search_improve(sel, reach, deadline)
            evaluate_and_update(sel_ls, reach_ls)

    # 2. Randomized Greedy
    random.seed(42)
    
    # Run randomized greedy for the remaining time
    while time.time() < deadline - 10.0:
        shuffled = list(universe)
        random.shuffle(shuffled)
        sel, reach = greedy_select(shuffled)
        evaluate_and_update(sel, reach)
        
        # Occasionally apply local search to a random result if time permits
        if time.time() < deadline - 5.0 and best_solution is not None:
             # We only do local search on the current best if it's significantly better or randomly?
             # Let's just do local search on the current best at the end.
             pass

    # 3. Final Local Search on the Best Solution Found
    if best_solution is not None:
        # Re-compute reachable for the best solution to ensure consistency
        best_reachable = {0}
        for x in best_solution:
            new_sums = {s + x for s in best_reachable if s + x <= target}
            best_reachable.update(new_sums)
        
        best_solution, best_reachable = local_search_improve(best_solution, best_reachable, deadline)

    # Fallback
    if best_solution is None:
        best_solution, _ = greedy_select(list(range(100, 0, -1)))

    # Final Verification (Safety Check)
    # Verify that no subset sums to 200
    reachable = {0}
    valid = True
    for x in best_solution:
        new_sums = {s + x for s in reachable if s + x <= target}
        if target in new_sums:
            valid = False
            break
        reachable.update(new_sums)
    
    if not valid:
        # Fallback to a known good simple set if verification fails
        # The descending greedy is very likely valid
        fallback_sel, _ = greedy_select(list(range(100, 0, -1)))
        return list(fallback_sel)

    return list(best_solution)

# EVOLVE-BLOCK-END
