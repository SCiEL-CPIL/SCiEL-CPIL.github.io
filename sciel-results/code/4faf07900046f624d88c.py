"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def _has_4ap(selected, x):
    """Check if adding x to selected creates a 4-term AP."""
    # For each possible common difference d, check if x-d, x-2d, x-3d are in selected
    # or x, x-d, x-2d, x-3d form a 4-AP with x as one endpoint
    # A 4-AP is: a, a+d, a+2d, a+3d
    # If x is the last term: x-3d, x-2d, x-d must be in selected
    # If x is the third term: x-2d, x-d, x+d must be in selected (but x+d not yet added)
    # If x is the second term: x-d, x+d, x+2d must be in selected (but x+d, x+2d not yet added)
    # If x is the first term: x+d, x+2d, x+3d must be in selected (but none yet added)
    
    # Since we're adding x to an existing set, we only need to check if x completes a 4-AP
    # where the other 3 terms are already in selected.
    
    # Case 1: x is the 4th term: x-3d, x-2d, x-d in selected
    for d in range(1, (x-1)//3 + 1):
        if all(x - j*d in selected for j in (1, 2, 3)):
            return True
    
    # Case 2: x is the 3rd term: x-2d, x-d, x+d in selected
    for d in range(1, (x-1)//2 + 1):
        if (x - 2*d in selected) and (x - d in selected) and (x + d in selected):
            return True
    
    # Case 3: x is the 2nd term: x-d, x+d, x+2d in selected
    for d in range(1, (x-1)//1 + 1):  # d < x
        if (x - d in selected) and (x + d in selected) and (x + 2*d in selected):
            return True
    
    # Case 4: x is the 1st term: x+d, x+2d, x+3d in selected
    for d in range(1, (512 - x)//3 + 1):
        if all(x + j*d in selected for j in (1, 2, 3)):
            return True
    
    return False

def _greedy_construct():
    selected = set()
    for x in range(1, 513):
        if not _has_4ap(selected, x):
            selected.add(x)
    return selected

def _local_search(selected, max_iter=50000):
    """Improve the set using swap moves: remove one element, add two others."""
    selected = set(selected)
    current_size = len(selected)
    
    for _ in range(max_iter):
        improved = False
        # Try removing each element and adding two others
        for rem in list(selected):
            # Skip if removing would leave us worse
            trial = selected - {rem}
            # Try adding pairs
            for add1 in range(1, 513):
                if add1 in trial:
                    continue
                trial_with_1 = trial | {add1}
                if _has_4ap(trial, add1):
                    continue
                for add2 in range(add1+1, 513):
                    if add2 in trial_with_1:
                        continue
                    if _has_4ap(trial_with_1, add2):
                        continue
                    new_set = trial_with_1 | {add2}
                    if len(new_set) > current_size:
                        selected = new_set
                        current_size = len(new_set)
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return selected

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 5s safety margin
    
    # Start with greedy construction
    best_set = _greedy_construct()
    best_size = len(best_set)
    
    # Try local search
    if time.time() < deadline - 10:
        improved = _local_search(best_set, max_iter=10000)
        if len(improved) > best_size:
            best_set = improved
            best_size = len(best_set)
    
    return sorted(best_set)
# EVOLVE-BLOCK-END
