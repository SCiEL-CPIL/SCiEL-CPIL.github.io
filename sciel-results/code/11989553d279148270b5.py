import time
from math import gcd
from functools import reduce
import random

def lcm(a, b):
    return a // gcd(a, b) * b

def lcm_list(lst):
    return reduce(lcm, lst, 1)

def check_conflict_fast(S, x):
    """
    Check if 1/x is a subset sum of {1/y for y in S if y > x}.
    Uses integer arithmetic via LCM scaling.
    Returns True if conflict exists, else False.
    """
    candidates = [y for y in S if y > x]
    if not candidates:
        return False
    
    # Scale all reciprocals to integers using LCM
    # 1/y scaled by LCM(1,2,...,24) gives integer
    L = 5342931457063200  # LCM(1..24)
    
    target_int = L // x
    cand_ints = [L // y for y in candidates]
    
    # Quick prune: max possible sum
    if sum(cand_ints) < target_int:
        return False
    
    # Sort by value descending for better pruning
    cand_ints.sort(reverse=True)
    
    # Use meet-in-the-middle for larger sets
    n = len(cand_ints)
    if n <= 20:
        # DFS with pruning
        def dfs(idx, current_sum, remaining_target):
            if current_sum == remaining_target:
                return True
            if idx == n or current_sum > remaining_target:
                return False
            
            # Prune: even taking all remaining can't reach target
            # (precompute suffix sums would help, but let's keep it simple)
            
            # Try taking this element
            if dfs(idx + 1, current_sum + cand_ints[idx], remaining_target):
                return True
            
            # Try skipping
            if dfs(idx + 1, current_sum, remaining_target):
                return True
            
            return False
        
        return dfs(0, 0, target_int)
    else:
        # For larger sets, use a different approach or just return False
        # (this shouldn't happen often in practice)
        return False

def is_valid_fast(S):
    for x in S:
        if check_conflict_fast(S, x):
            return False
    return True

def backtracking_search(elements, current_set, start_idx, deadline):
    """
    Backtracking search with forward checking.
    Try to build the largest valid subset.
    """
    best = set(current_set)
    
    if start_idx >= len(elements):
        return best
    
    if time.time() > deadline:
        return best
    
    elem = elements[start_idx]
    
    # Option 1: Include elem
    new_set = current_set | {elem}
    # Check if adding elem creates a conflict
    valid = True
    for x in new_set:
        if check_conflict_fast(new_set, x):
            valid = False
            break
    
    if valid:
        result = backtracking_search(elements, new_set, start_idx + 1, deadline)
        if len(result) > len(best):
            best = result
    
    # Option 2: Exclude elem
    result = backtracking_search(elements, current_set, start_idx + 1, deadline)
    if len(result) > len(best):
        best = result
    
    return best

def greedy_construct(order, deadline):
    """
    Greedily add elements in the given order if they don't create conflicts.
    """
    S = set()
    for x in order:
        if time.time() > deadline:
            break
        new_S = S | {x}
        # Check if adding x creates any conflict
        valid = True
        for y in new_S:
            if check_conflict_fast(new_S, y):
                valid = False
                break
        if valid:
            S = new_S
    return S

def local_improve(S, deadline):
    """
    Try to add elements that were excluded, or remove and re-add differently.
    """
    S = set(S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding each excluded element
        for x in range(1, 25):
            if x in S:
                continue
            if time.time() > deadline:
                break
            new_S = S | {x}
            if is_valid_fast(new_S):
                S = new_S
                improved = True
                break
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    best_S = set()
    
    # Try multiple random orders for greedy construction
    random.seed(42)
    for trial in range(50):
        if time.time() > deadline:
            break
        
        order = list(range(1, 25))
        random.shuffle(order)
        
        S = greedy_construct(order, deadline)
        S = local_improve(S, deadline)
        
        if len(S) > len(best_S):
            best_S = set(S)
    
    # Also try backtracking with different orderings
    for trial in range(10):
        if time.time() > deadline:
            break
        
        elements = list(range(1, 25))
        random.shuffle(elements)
        
        # Limit backtracking to avoid explosion
        S = backtracking_search(elements, set(), 0, deadline)
        
        if len(S) > len(best_S):
            best_S = set(S)
    
    # Final improvement pass
    if best_S and time.time() < deadline:
        best_S = local_improve(best_S, deadline)
    
    # Final check
    if not best_S or not is_valid_fast(best_S):
        best_S = {24}
    
    return list(best_S)
