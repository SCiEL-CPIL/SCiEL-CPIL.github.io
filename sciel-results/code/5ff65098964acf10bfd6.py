import time
import math
import random

_START = time.time()
_DEADLINE = 85.0

def _time_left():
    return _DEADLINE - (time.time() - _START)

N = 24
_L = 1
for k in range(1, 25):
    _L = _L * k // math.gcd(_L, k)
_W = [_L // k for k in range(1, 25)]

def _subset_sums(elements):
    """Returns the set of all subset sums of elements."""
    sums = {0}
    for x in elements:
        new_sums = {s + x for s in sums}
        sums.update(new_sums)
    return sums

def _is_valid_addition(k_idx, selected_indices):
    """
    Check if adding k_idx to selected_indices is valid.
    selected_indices does not contain k_idx.
    """
    S = selected_indices
    w_k = _W[k_idx]
    
    # 1. Check if w_k is a subset sum of S
    if S:
        sums_S = _subset_sums([_W[i] for i in S])
        if w_k in sums_S:
            return False
    
    # 2. Check if any y in S becomes subsumed
    for y in S:
        w_y = _W[y]
        # Check if w_y is a subset sum of (S \ {y}) U {k_idx}
        # This is true if w_y in sums_{S\{y}} (False, assumed valid before) 
        # OR w_y - w_k in sums_{S\{y}}
        diff = w_y - w_k
        if diff < 0:
            continue
        # Need to check if diff is in sums_{S \ {y}}
        others = [i for i in S if i != y]
        if not others:
            if diff == 0:
                return False
            else:
                continue
        sums_others = _subset_sums([_W[i] for i in others])
        if diff in sums_others:
            return False
            
    return True

def _validate_set(indices):
    """Full validation of a set of indices."""
    S = set(indices)
    for x in S:
        others = [i for i in S if i != x]
        if not others:
            continue
        sums_others = _subset_sums([_W[i] for i in others])
        if _W[x] in sums_others:
            return False
    return True

def _local_search(current_set):
    """
    Improve current_set by trying to add elements, and if stuck, by swapping.
    Returns a potentially larger valid set.
    """
    current = list(current_set)
    current_set_obj = set(current)
    improved = True
    while improved and _time_left() > 5.0:
        improved = False
        
        # Phase 1: Try to add any missing element
        for k in range(24):
            if k in current_set_obj:
                continue
            if _is_valid_addition(k, current):
                current.append(k)
                current_set_obj.add(k)
                improved = True
                break # Restart scan as set changed
        
        if improved:
            continue
            
        # Phase 2: Try swaps (remove 1, add 1)
        # This helps escape local maxima where adding is impossible but swapping yields a better configuration for future additions
        swap_found = False
        for i in range(len(current)):
            if _time_left() < 5.0:
                break
            remove_idx = current[i]
            # Try adding any missing element k, then check if we can remove remove_idx?
            # Actually, standard swap: remove r, add a. Check if valid.
            # But we want to maximize size. A 1-for-1 swap doesn't change size.
            # We should look for 1-for-2 swaps (remove 1, add 2) to increase size.
            
            # Let's try 1-for-2 swaps: remove r, add a, add b.
            # This is O(N^3) which is 24^3 ~ 13k checks. Each check is O(2^N). Might be slow.
            # Let's stick to 1-for-1 swaps to "unblock" the set, allowing subsequent additions.
            
            # Try 1-for-1 swap
            for a in range(24):
                if a in current_set_obj:
                    continue
                # Check if removing remove_idx and adding a is valid
                # New set: (current - {remove_idx}) U {a}
                # We need to check validity of a in (current - {remove_idx})
                # AND that no other element in (current - {remove_idx}) is subsumed by the new set.
                
                # This is complex to check incrementally. 
                # Simpler: Construct new list, validate fully? 
                # Full validation is O(N * 2^N). N=12, 2^12=4096. 12*4096 ~ 50k ops.
                # 24*24 swaps = 576 swaps. 576 * 50k = 28M ops. Feasible in Python if optimized? 
                # Maybe too slow.
                
                # Let's try a simpler heuristic: 
                # Just try to add. If we can't, try removing the "most conflicting" element 
                # and see if we can add more.
                
                pass
        
        # Alternative Local Search: Random Restart / Shake
        # If we are stuck, perturb the set by removing 1-2 random elements and re-running greedy fill.
        if _time_left() > 10.0:
             # Perturb
             if current:
                 # Remove a random element
                 rem_idx = random.choice(current)
                 current.remove(rem_idx)
                 current_set_obj.discard(rem_idx)
                 # Re-run greedy fill on the perturbed set
                 for k in range(24):
                     if k in current_set_obj:
                         continue
                     if _is_valid_addition(k, current):
                         current.append(k)
                         current_set_obj.add(k)
                 improved = True # Force loop to continue
        
    return current

def construct():
    best_set = []
    
    # Try multiple orders for initial greedy
    orders = [
        list(range(23, -1, -1)), # Descending index (24 to 1)
        list(range(24)),         # Ascending index (1 to 24)
    ]
    
    random.seed(42)
    for _ in range(50):
        order = list(range(24))
        random.shuffle(order)
        orders.append(order)
        
    for order in orders:
        if _time_left() < 10.0:
            break
            
        current = []
        for k in order:
            if _time_left() < 2.0:
                break
            if _is_valid_addition(k, current):
                current.append(k)
                
        # Local search
        current = _local_search(current)
        
        if len(current) > len(best_set):
            best_set = current
            
    # Final verification
    if not best_set:
        return [24]
        
    # Ensure validity
    if not _validate_set(best_set):
        # Fallback to a known safe set (large numbers)
        fallback = list(range(12, 24)) # 13..24
        if _validate_set(fallback):
            return [i + 1 for i in fallback]
        return [24]
        
    return [i + 1 for i in best_set]

