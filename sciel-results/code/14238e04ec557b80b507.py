import random
import time
import sys
from itertools import combinations

L = 5354228880  # lcm(1..24)

def subset_sum_exists(weights, target):
    """
    Check if target can be formed by a subset of weights.
    Uses meet-in-the-middle.
    Returns True if found, False otherwise.
    """
    n = len(weights)
    if n == 0:
        return target == 0
    if target == 0:
        return True
    
    # Filter weights > target
    weights = [w for w in weights if w <= target]
    if not weights:
        return target == 0
        
    n = len(weights)
    mid = n // 2
    left = weights[:mid]
    right = weights[mid:]
    
    left_sums = set()
    left_sums.add(0)
    for w in left:
        new_sums = set()
        for s in left_sums:
            if s + w <= target:
                new_sums.add(s + w)
        left_sums.update(new_sums)
        
    right_sums = set()
    right_sums.add(0)
    for w in right:
        new_sums = set()
        for s in right_sums:
            if s + w <= target:
                new_sums.add(s + w)
        right_sums.update(new_sums)
        
    # Check intersection
    for s in left_sums:
        if (target - s) in right_sums:
            return True
    return False

def is_valid(S):
    """Check if set S is valid (no reciprocal subsum violations)."""
    S_list = list(S)
    for x in S_list:
        target = L // x
        weights = [L // y for y in S_list if y != x]
        if subset_sum_exists(weights, target):
            return False
    return True

def backtracking_search(elements, current, remaining, deadline):
    """
    Backtracking search with forward checking to find largest valid subset.
    elements: list of all elements
    current: set of currently selected elements
    remaining: list of elements not yet considered
    deadline: time limit
    Returns best set found.
    """
    best = set(current)
    best_size = len(current)
    
    if not remaining:
        return best
    
    if time.time() > deadline:
        return best
    
    # Try including remaining[0]
    x = remaining[0]
    rest = remaining[1:]
    
    # Check if adding x keeps current valid
    test_set = current | {x}
    if is_valid(test_set):
        # Forward checking: prune if x creates conflicts that make future additions impossible
        # For now, just recurse
        result = backtracking_search(elements, test_set, rest, deadline)
        if len(result) > best_size:
            best = result
            best_size = len(result)
    
    # Try excluding x
    result = backtracking_search(elements, current, rest, deadline)
    if len(result) > best_size:
        best = result
        best_size = len(result)
    
    return best

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin
    
    all_elements = list(range(1, 25))
    
    # Start with a greedy construction to get a good initial set
    S = set()
    for x in all_elements:
        if time.time() > deadline:
            break
        test_set = S | {x}
        if is_valid(test_set):
            S = test_set
    
    best_set = set(S)
    best_size = len(S)
    
    # Now try to improve by backtracking search on the full space
    # But first, let's try local search: remove some elements and re-add
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Start from current best or a random subset
        if restart == 0:
            S = set(best_set)
        else:
            # Random perturbation: start from best_set and randomly remove 1-3 elements
            S = set(best_set)
            to_remove = min(len(S), random.randint(1, 3))
            for _ in range(to_remove):
                if S:
                    S.discard(random.choice(list(S)))
        
        # Try to expand S by adding elements
        missing = [x for x in all_elements if x not in S]
        random.shuffle(missing)
        
        for m in missing:
            if time.time() > deadline:
                break
            S_test = S | {m}
            if is_valid(S_test):
                S = S_test
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
        
        # If we found a very good set, try to polish it
        if best_size >= 18:
            # Try all single removals and re-additions
            improved = True
            while improved and time.time() < deadline:
                improved = False
                for x in list(best_set):
                    if time.time() > deadline:
                        break
                    S_test = best_set - {x}
                    # Try to add back any missing element
                    for m in [y for y in all_elements if y not in S_test]:
                        if time.time() > deadline:
                            break
                        S_new = S_test | {m}
                        if is_valid(S_new) and len(S_new) > best_size:
                            best_set = set(S_new)
                            best_size = len(S_new)
                            improved = True
                            break
                    if improved:
                        break
    
    if not best_set:
        return [1]
        
    return list(best_set)
