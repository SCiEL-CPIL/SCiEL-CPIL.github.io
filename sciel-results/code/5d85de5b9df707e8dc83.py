import random
import numpy as np

N = 256

def _is_sidon(s):
    """Check if set s is a Sidon set (all pairwise sums a+b with a<=b are distinct)."""
    sums = set()
    for i in range(len(s)):
        for j in range(i, len(s)):
            sm = s[i] + s[j]
            if sm in sums:
                return False
            sums.add(sm)
    return True

def _is_maximal(s, n):
    """Check if Sidon set s is inclusion-maximal in [1, n]."""
    if not s:
        return False
    s_set = set(s)
    sums = set()
    for i in range(len(s)):
        for j in range(i, len(s)):
            sums.add(s[i] + s[j])
    
    for x in range(1, n + 1):
        if x in s_set:
            continue
        # Check if adding x would break Sidon property
        # If adding x breaks Sidon, x cannot be added, which is good for maximality.
        # If adding x does NOT break Sidon, then x CAN be added, so s is NOT maximal.
        
        # Optimization: Check conflicts quickly
        conflict = False
        for a in s:
            if (x + a) in sums:
                conflict = True
                break
        if conflict:
            continue
        
        if (2 * x) in sums:
            conflict = True
            continue
            
        # Check if x+x or x+a conflicts with other new sums?
        # Since we only add one element x, the new sums are x+x and x+a for a in s.
        # We need to ensure these new sums don't conflict with each other or existing sums.
        # We already checked against existing sums.
        # Do new sums conflict with each other?
        # x+a1 == x+a2 => a1==a2 (distinct in s, so no)
        # x+a1 == 2x => a1 == x (x not in s, so no)
        # So if no conflict with existing sums, x can be added.
        
        if not conflict:
            return False
            
    return True

def _greedy_maximal(order):
    """Build a maximal Sidon set by processing elements in given order."""
    selected = []
    sums = set()
    for x in order:
        added = [x + a for a in selected] + [2 * x]
        # Check internal conflicts of new sums
        if len(set(added)) != len(added):
            continue
        # Check against existing sums
        if not sums.intersection(added):
            selected.append(x)
            sums.update(added)
    
    # Ensure maximality by adding any remaining valid elements
    s_set = set(selected)
    changed = True
    while changed:
        changed = False
        for x in range(1, N + 1):
            if x in s_set:
                continue
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
                s_set.add(x)
                changed = True
    return selected

def _local_search(s):
    """
    Improve a maximal Sidon set by removing redundant elements or swapping.
    Returns a potentially smaller maximal Sidon set.
    """
    current = list(s)
    improved = True
    while improved:
        improved = False
        # Try to remove elements
        for i in range(len(current) - 1, -1, -1):
            candidate = current[i]
            new_set = current[:i] + current[i+1:]
            if _is_sidon(new_set) and _is_maximal(new_set, N):
                current = new_set
                improved = True
                break
        
        if improved:
            continue
            
        # Try swaps: replace x with y
        # We look for y that allows removal of x and addition of y to maintain maximality
        # and potentially reduce size? No, swap keeps size same.
        # But maybe swapping allows subsequent removals?
        # Let's try simple swaps first to escape local minima.
        s_set = set(current)
        for i in range(len(current)):
            x = current[i]
            # Try replacing x with a smaller number y not in set
            # Sort candidates by value to prefer smaller numbers (often better for packing)
            candidates = [y for y in range(1, N + 1) if y not in s_set and y < x]
            random.shuffle(candidates)
            
            for y in candidates[:5]: # Limit candidates for speed
                new_set = current[:]
                new_set[i] = y
                if _is_sidon(new_set) and _is_maximal(new_set, N):
                    current = new_set
                    improved = True
                    break
            if improved:
                break
                
    return current

def construct():
    best_set = None
    best_size = N + 1
    
    # Strategy: Multiple random restarts with greedy construction + local search
    for restart in range(100):
        # Create a random order
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        # Greedy construction
        current = _greedy_maximal(order)
        
        # Local search to minimize size
        current = _local_search(current)
        
        if len(current) < best_size:
            best_size = len(current)
            best_set = list(current)
            
            # Early termination if we hit a known lower bound (e.g., ~16 for N=256)
            if best_size <= 16:
                break

    # Fallback if no solution found (should not happen)
    if best_set is None:
        order = list(range(1, N + 1))
        best_set = _greedy_maximal(order)
        best_set = _local_search(best_set)
        
    return best_set
