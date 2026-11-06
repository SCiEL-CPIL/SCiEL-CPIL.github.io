import time
import random
import math

def _is_dissociated(S):
    """
    Check if all subset sums of {1/s for s in S} are distinct.
    Returns True if dissociated, False otherwise.
    """
    if len(S) <= 1:
        return True
    
    # Calculate LCM
    L = 1
    for s in S:
        L = L * s // math.gcd(L, s)
        # If L is extremely large, we might run into performance issues with big ints,
        # but for n<=22 and k<=15, L is manageable in Python.
        
    # Convert to integer weights
    weights = [L // s for s in S]
    
    # Check distinct subset sums
    seen = {0}
    current_sums = [0]
    for w in weights:
        new_sums = [x + w for x in current_sums]
        for ns in new_sums:
            if ns in seen:
                return False
        seen.update(new_sums)
        current_sums.extend(new_sums)
        
    return True

def _try_expand(S, all_elements, deadline):
    """
    Try to greedily expand set S by adding elements from all_elements 
    that are not in S, maintaining the dissociated property.
    """
    current = set(S)
    improved = True
    while improved:
        improved = False
        for e in all_elements:
            if e in current:
                continue
            if time.time() > deadline:
                return current
            new_set = current | {e}
            if _is_dissociated(new_set):
                current = new_set
                improved = True
                break # Restart the loop to check all elements again with the new set
    return current

def _try_swap_expand(S, all_elements, deadline):
    """
    Try to find a swap (remove one, add one) that allows for further expansion.
    """
    current = set(S)
    current_list = list(current)
    improved = False
    
    for i, remove_e in enumerate(current_list):
        if time.time() > deadline:
            return current, improved
        # Try adding any element not in current
        for add_e in all_elements:
            if add_e in current:
                continue
            new_set = (current - {remove_e}) | {add_e}
            if _is_dissociated(new_set):
                # Try to expand this new set
                expanded = _try_expand(new_set, all_elements, deadline)
                if len(expanded) > len(current):
                    return expanded, True
    
    return current, improved

def construct():
    start = time.time()
    deadline = start + 55.0 # Safety margin for 60s limit
    
    all_elements = list(range(1, 23))
    best_set = set()
    best_size = 0
    
    # Phase 1: Deterministic Greedy
    orders = [
        all_elements,
        sorted(all_elements, reverse=True),
        sorted(all_elements, key=lambda x: math.gcd(x, 22)), # Heuristic: share factors with max?
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        current = set()
        for e in order:
            if time.time() > deadline:
                break
            new_set = current | {e}
            if _is_dissociated(new_set):
                current = new_set
        
        if len(current) > best_size:
            best_size = len(current)
            best_set = set(current)
            
        # Expand the greedily built set
        expanded = _try_expand(current, all_elements, deadline)
        if len(expanded) > best_size:
            best_size = len(expanded)
            best_set = set(expanded)

    # Phase 2: Random Search
    rng = random.Random(42)
    
    # Try random subsets of various sizes
    for _ in range(500):
        if time.time() > deadline - 5.0:
            break
            
        # Pick a random size to start with, between 4 and 10
        k = rng.randint(4, 10)
        random_subset = set(rng.sample(all_elements, k))
        
        # Ensure it's dissociated by removing elements if necessary
        # Simple approach: try to add/remove until valid
        # Or just check if valid. If not, skip or try to repair.
        # Let's try to repair by removing elements that cause collisions.
        # This is complex. Instead, let's just try random subsets and if valid, expand.
        
        if _is_dissociated(random_subset):
            expanded = _try_expand(random_subset, all_elements, deadline)
            if len(expanded) > best_size:
                best_size = len(expanded)
                best_set = set(expanded)
        else:
            # Try to repair: remove one random element and see if it becomes valid
            # This is inefficient. Better to just generate valid sets.
            pass

    # Phase 3: Local Search on Best Set
    if best_set:
        # Try to add any remaining element
        expanded = _try_expand(best_set, all_elements, deadline)
        if len(expanded) > best_size:
            best_size = len(expanded)
            best_set = set(expanded)
            
        # Try swaps
        # We want to find a swap that leads to a larger set.
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            current_set = set(best_set)
            
            for remove_e in list(current_set):
                if time.time() > deadline - 2.0:
                    break
                for add_e in all_elements:
                    if add_e in current_set:
                        continue
                    new_set = (current_set - {remove_e}) | {add_e}
                    if _is_dissociated(new_set):
                        # Try to expand
                        expanded = _try_expand(new_set, all_elements, deadline)
                        if len(expanded) > len(best_set):
                            best_set = set(expanded)
                            best_size = len(best_set)
                            improved = True
                            break
                if improved:
                    break

    if not best_set:
        # Fallback to a known valid set
        best_set = {1, 2, 3, 5, 7, 11, 13, 17, 19}
        
    return list(best_set)
