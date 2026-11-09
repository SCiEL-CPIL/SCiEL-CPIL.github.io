import time
import random
from math import lcm
from functools import reduce

UNIVERSE = list(range(1, 25))
L = reduce(lcm, UNIVERSE)

# Precompute weights
WEIGHTS = {x: L // x for x in UNIVERSE}

def get_conflicts(S):
    """
    Returns a list of conflicts.
    A conflict is a tuple (x, T) where x in S and T is a subset of S\{x}
    such that sum(1/y for y in T) == 1/x.
    In weight terms: w_x == sum(w_y for y in T).
    """
    conflicts = []
    S_list = list(S)
    n = len(S_list)
    if n == 0:
        return conflicts
    
    weights = [WEIGHTS[x] for x in S_list]
    
    for i, x in enumerate(S_list):
        target = weights[i]
        others = weights[:i] + weights[i+1:]
        
        if not others:
            continue
            
        # Check if target is in subset sums of others
        # Use meet-in-the-middle for speed if n is large, but n <= 24.
        # For n=24, 2^12 = 4096. Very fast.
        m = len(others)
        mid = m // 2
        left = others[:mid]
        right = others[mid:]
        
        left_sums = {0}
        for w in left:
            left_sums = left_sums | {s + w for s in left_sums}
            
        found = False
        # We just need to know if it's found, not the exact subset, 
        # but for removal strategy, knowing x is sufficient.
        right_sums = {0}
        for w in right:
            new_sums = {s + w for s in right_sums}
            for s in new_sums:
                if target - s in left_sums:
                    found = True
                    break
            if found:
                break
            right_sums |= new_sums
            
        if found:
            conflicts.append(x)
            
    return conflicts

def is_valid(S):
    return len(get_conflicts(S)) == 0

def construct():
    deadline = time.time() + 85  # 5s safety margin
    
    # Start with the full set
    current = set(UNIVERSE)
    
    # Phase 1: Greedy removal of conflicts
    # We want to keep as many elements as possible.
    # If x is in conflict, we can remove x.
    # Or we can remove an element that forms the sum.
    # Removing x is a valid move.
    
    while time.time() < deadline:
        conflicts = get_conflicts(current)
        if not conflicts:
            break
        
        # Pick a conflict to resolve
        # Randomize to escape local minima? 
        # Or pick the one that seems "most central"?
        # Simple: remove the first conflict found, or random one.
        # Let's try removing the element x from the conflict.
        # To maximize size, maybe we should remove the element that is involved in the most conflicts?
        # But computing "most involved" is expensive.
        
        # Heuristic: Remove the smallest element in the conflict? 
        # Or the largest?
        # Let's just pick a random conflict and remove x.
        x_to_remove = random.choice(conflicts)
        current.remove(x_to_remove)
        
    # Phase 2: Try to add back removed elements or swap
    # The set of removed elements
    removed = set(UNIVERSE) - current
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding one back
        for x in list(removed):
            if time.time() > deadline:
                break
            candidate = current | {x}
            if is_valid(candidate):
                current = candidate
                removed.remove(x)
                improved = True
        
        # Try swapping: remove one from current, add two from removed?
        # Or remove one from current, add one from removed if size increases? 
        # Size doesn't increase with 1-for-1 swap.
        # We want to increase size.
        # So we need to remove 1 and add 2? Or just add 1 if possible (handled above).
        
        # If we can't add any single element, try to swap 1 in current with 1 in removed? 
        # This keeps size same, but might allow adding another one later.
        # Let's try simple local search: 
        # Remove one element from current, see if we can add 2 from removed.
        
        if not improved:
             # Try 1-out 2-in
             for x_in_current in list(current):
                 if time.time() > deadline:
                     break
                 temp_current = current - {x_in_current}
                 # Try to add 2 from removed
                 added = False
                 for y in removed:
                     if time.time() > deadline:
                         break
                     for z in removed:
                         if y == z:
                             continue
                         candidate = temp_current | {y, z}
                         if is_valid(candidate):
                             current = candidate
                             removed.remove(y)
                             removed.remove(z)
                             added = True
                             break
                     if added:
                         break
                 if added:
                     improved = True
                     break
                     
    return list(current)
