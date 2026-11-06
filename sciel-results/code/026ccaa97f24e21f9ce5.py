"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _greedy(order):
    selected, reachable = [], {0}
    for x in order:
        new = {s + x for s in reachable if s + x <= 200}
        if 200 not in new:
            selected.append(x)
            reachable |= new
    return selected

def _try_add_back(selected, reachable, excluded):
    """Try to add back excluded elements one by one."""
    selected_set = set(selected)
    for x in excluded:
        if x in selected_set:
            continue
        # Check if adding x creates a subset sum of 200
        # We need to check if there's a subset of current selected that sums to 200-x
        # Use the reachable set: if (200-x) is in reachable, then adding x would create 200
        if x >= 1 and x <= 100 and (200 - x) not in reachable:
            # Also need to check: does adding x create new sums that, with existing elements, make 200?
            # Actually if 200-x is in reachable, then some subset sums to 200-x, and adding x gives 200.
            # If 200-x is NOT in reachable, adding x is safe w.r.t. the current set.
            # But we also need to make sure that new sums created don't combine with other elements to make 200.
            # The reachable set tracks all subset sums of selected. Adding x creates new sums s+x for s in reachable.
            # We need 200 not to be in {s+x for s in reachable if s+x <= 200}
            new_sums = {s + x for s in reachable if s + x <= 200}
            if 200 not in new_sums:
                selected.append(x)
                selected_set.add(x)
                reachable |= new_sums
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s deadline, 10s safety margin
    
    best = []
    best_len = 0
    
    # Try descending order (original)
    order_desc = list(range(100, 0, -1))
    result = _greedy(order_desc)
    if len(result) > best_len:
        best = result
        best_len = len(result)
    
    # Try ascending order
    if time.time() < deadline:
        order_asc = list(range(1, 101))
        result = _greedy(order_asc)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Try several random permutations
    random.seed(42)
    for i in range(50):
        if time.time() >= deadline:
            break
        order = list(range(1, 101))
        random.shuffle(order)
        result = _greedy(order)
        if len(result) > best_len:
            best = result
            best_len = len(result)
    
    # Local search: try to add back excluded elements
    if time.time() < deadline:
        best_set = set(best)
        excluded = [x for x in range(1, 101) if x not in best_set]
        # Rebuild reachable from best
        reachable = {0}
        for x in best:
            reachable |= {s + x for s in list(reachable) if s + x <= 200}
        improved = _try_add_back(list(best), reachable, excluded)
        if len(improved) > best_len:
            best = improved
            best_len = len(improved)
    
    return best if best else [100]
# EVOLVE-BLOCK-END
