"""
Rewritten program: Backtracking search with exact rational arithmetic 
to find the maximum dissociated reciprocal set.
"""
import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin

    universe = list(range(1, 23))
    
    best_set = []
    best_size = 0
    
    # We will use backtracking.
    # To maximize the chance of finding a large set quickly, 
    # we can try orderings that might yield larger sets.
    # However, backtracking is exhaustive for the given order.
    # The set of all valid subsets is independent of order.
    # So one exhaustive search is enough, but we need to prune well.
    
    # Let's use descending order (22..1).
    # Why? Small reciprocals.
    # Actually, let's try both and take the best.
    
    orderings = [
        list(range(22, 0, -1)),
        list(range(1, 23))
    ]
    
    def dfs(current_set, current_sums, remaining_idx):
        nonlocal best_set, best_size, deadline, start_time
        
        if time.time() > deadline:
            return

        # Pruning
        if len(current_set) + len(remaining_idx) <= best_size:
            return

        if not remaining_idx:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
            return

        elem = remaining_idx[0]
        rest = remaining_idx[1:]
        
        # Branch 1: Exclude elem
        dfs(current_set, current_sums, rest)
        
        # Branch 2: Include elem
        r = Fraction(1, elem)
        
        # Check validity
        # If current_sums is {0}, valid.
        # Else, check if (s + r) is in current_sums for any s in current_sums.
        
        valid = True
        if len(current_sums) > 1:
            for s in current_sums:
                if (s + r) in current_sums:
                    valid = False
                    break
        
        if valid:
            # Create new sums
            # Optimization: if len(current_sums) is small, this is fast.
            new_sums = current_sums | {s + r for s in current_sums}
            dfs(current_set + [elem], new_sums, rest)

    for ordering in orderings:
        if time.time() > deadline:
            break
        
        initial_sums = {Fraction(0)}
        dfs([], initial_sums, ordering)
        
        # If we found a large set, we might stop.
        # Max possible is 22, but realistically < 12.
        if best_size >= 11:
            break

    if not best_set:
        # Fallback
        best_set = [2, 3, 5, 7, 11]
        
    return sorted(best_set)
