import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    
    def is_valid(S):
        """Check if set S has no subset summing to 200."""
        # Use DP for subset sum up to 200
        reachable = [False] * 201
        reachable[0] = True
        for x in S:
            if x > 200:
                continue
            for s in range(200 - x, -1, -1):
                if reachable[s]:
                    reachable[s + x] = True
                    if s + x == 200:
                        return False
        return not reachable[200]
    
    def can_add(S, x, reachable_arr):
        """Check if adding x to S keeps it valid, using precomputed reachable array."""
        if x > 200:
            return False
        # Check if 200 - x is reachable in current S
        if 200 - x >= 0 and reachable_arr[200 - x]:
            return False
        # Also need to check that adding x doesn't create 200 via other combinations
        # Actually the above check is sufficient: if 200-x is reachable, adding x makes 200 reachable.
        # If 200-x is not reachable, then no subset of S sums to 200-x, so no subset of S∪{x} sums to 200
        # (since any such subset either doesn't use x, in which case it's a subset of S summing to 200 which we already avoid,
        #  or uses x, in which case the rest sums to 200-x which is not reachable)
        return True
    
    def update_reachable(reachable_arr, x):
        """Update reachable array after adding x."""
        for s in range(200 - x, -1, -1):
            if reachable_arr[s]:
                reachable_arr[s + x] = True
    
    # Start with the known good set {67, 68, ..., 100}
    S = set(range(67, 101))
    reachable = [False] * 201
    reachable[0] = True
    for x in sorted(S):
        update_reachable(reachable, x)
    
    # Try to add elements from 66 down to 1
    for x in range(66, 0, -1):
        if time.time() > deadline:
            break
        if can_add(S, x, reachable):
            S.add(x)
            update_reachable(reachable, x)
    
    # Try random restarts with different initial sets
    best_S = S
    best_size = len(S)
    
    # Also try greedy from different orderings
    def greedy_construction(order):
        selected = set()
        reach = [False] * 201
        reach[0] = True
        for x in order:
            if can_add(selected, x, reach):
                selected.add(x)
                update_reachable(reach, x)
        return selected
    
    # Try several orderings
    orderings = [
        list(range(100, 0, -1)),
        list(range(1, 101)),
        list(range(67, 101)) + list(range(66, 0, -1)),
        list(range(66, 0, -1)) + list(range(67, 101)),
    ]
    
    for order in orderings:
        if time.time() > deadline:
            break
        S_test = greedy_construction(order)
        if len(S_test) > best_size:
            best_size = len(S_test)
            best_S = S_test
    
    # Local search: try to add more elements
    S = best_S
    reach = [False] * 201
    reach[0] = True
    for x in sorted(S):
        update_reachable(reach, x)
    
    missing = set(range(1, 101)) - S
    
    no_progress = 0
    max_no_progress = 50
    
    while time.time() < deadline and no_progress < max_no_progress:
        added = False
        # Try adding elements in descending order (larger elements are "safer" as they need smaller complements)
        for x in sorted(missing, reverse=True):
            if time.time() > deadline:
                break
            if can_add(S, x, reach):
                S.add(x)
                missing.discard(x)
                update_reachable(reach, x)
                added = True
                break
        
        if added:
            no_progress = 0
            continue
        
        # Try swaps: remove one, add one or more
        improved = False
        candidates = list(missing)
        random.shuffle(candidates)
        for x in candidates[:min(len(candidates), 30)]:
            if time.time() > deadline:
                break
            for y in list(S):
                if time.time() > deadline:
                    break
                S_test = S - {y}
                S_test.add(x)
                # Check validity
                reach_test = [False] * 201
                reach_test[0] = True
                valid = True
                for z in sorted(S_test):
                    if can_add(set(), z, reach_test):
                        update_reachable(reach_test, z)
                    else:
                        valid = False
                        break
                if not valid:
                    continue
                # Try to add more
                extra = 0
                for w in missing:
                    if w == x:
                        continue
                    if can_add(S_test, w, reach_test):
                        S_test.add(w)
                        update_reachable(reach_test, w)
                        extra += 1
                if len(S_test) > len(S):
                    S = S_test
                    reach = reach_test
                    missing = set(range(1, 101)) - S
                    improved = True
                    no_progress = 0
                    break
            if improved:
                break
        
        if not improved:
            no_progress += 1
    
    # Final polish
    reach = [False] * 201
    reach[0] = True
    for x in sorted(S):
        update_reachable(reach, x)
    missing = set(range(1, 101)) - S
    for x in sorted(missing, reverse=True):
        if can_add(S, x, reach):
            S.add(x)
            update_reachable(reach, x)
    
    return list(S)
