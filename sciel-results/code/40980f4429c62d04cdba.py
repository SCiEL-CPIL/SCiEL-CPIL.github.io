import time
import random
import numpy as np

_START_TIME = time.time()
_DEADLINE = 115.0  # Safety margin

def _elapsed():
    return time.time() - _START_TIME

def _can_add(counts, x, in_set):
    """Check if x can be added. Returns True if valid."""
    # Sums to check: x + y for y in set (y <= x is not required for count, just unique pairs a<=b)
    # The new pairs are (x, y) for all y in set with y != x (x not in set yet), and (x, x).
    # Actually, if x is not in set, the new pairs are (min(x,y), max(x,y)) for all y in set.
    # Since x is new, all these pairs are new.
    # We need counts[x+y] < 2 for all y in set, and counts[2*x] < 2.
    
    # Optimization: We can iterate through the set elements.
    # But iterating through 1000 elements for every check in a tight loop might be slow.
    # However, we only do this for candidates.
    
    # Let's use the fact that we have in_set.
    # We can iterate over indices where in_set is True.
    # This is O(N) per check.
    
    # Faster: Precomputed list of active elements?
    # For now, simple loop.
    
    for y in np.where(in_set)[0]:
        s = x + y
        if counts[s] >= 2:
            return False
    if counts[2 * x] >= 2:
        return False
    return True

def _add_element(counts, in_set, x, active_list):
    """Add x to the set, updating counts and structures."""
    in_set[x] = True
    active_list.append(x)
    for y in active_list:
        if y == x:
            continue
        s = x + y
        counts[s] += 1
    counts[2 * x] += 1

def _remove_element(counts, in_set, x, active_list):
    """Remove x from the set, updating counts and structures."""
    in_set[x] = False
    active_list.remove(x)
    for y in active_list:
        s = x + y
        counts[s] -= 1
    counts[2 * x] -= 1

def construct():
    global _START_TIME, _DEADLINE
    _START_TIME = time.time()
    random.seed(42)
    
    N = 1000
    max_sum = 2 * N
    counts = np.zeros(max_sum + 1, dtype=np.int32)
    in_set = np.zeros(N + 1, dtype=bool)
    active_list = []
    
    # 1. Greedy Initialization
    # Bias towards larger numbers? Or random?
    # Larger numbers have sums further apart, potentially less collision.
    # Let's try a mix: 50% random, 50% descending.
    
    candidates = list(range(1, N + 1))
    random.shuffle(candidates)
    
    # We will try to add elements.
    # To make it faster, we can just iterate through a specific order.
    # Order: Descending with noise.
    order = sorted(candidates, key=lambda x: -x + random.random() * 100)
    
    for x in order:
        if _elapsed() > _DEADLINE:
            break
        if _can_add(counts, x, in_set):
            _add_element(counts, in_set, x, active_list)
            
    best_size = len(active_list)
    best_set = list(active_list)
    
    # 2. Local Search
    # We try to improve the set.
    # Loop:
    #   - Try to add any unselected element.
    #   - If blocked, try to swap.
    
    # To speed up _can_add, we can keep a list of active elements.
    # We already have active_list.
    
    # We need to identify blocked candidates efficiently.
    # Instead of checking all N elements every time, we can maintain a list of "unselected" elements.
    unselected = [x for x in range(1, N + 1) if not in_set[x]]
    
    while _elapsed() < _DEADLINE - 1.0:
        progress = False
        
        # Pass 1: Try to add any unselected element
        # We iterate through a random subset of unselected to avoid bias?
        # Or just all of them. N=1000, so it's small.
        
        # Shuffle unselected to randomize which one we try to fix first
        random.shuffle(unselected)
        
        for x in unselected:
            if _elapsed() > _DEADLINE - 1.0:
                break
            
            if _can_add(counts, x, in_set):
                _add_element(counts, in_set, x, active_list)
                unselected.remove(x)
                progress = True
                best_size = len(active_list)
                best_set = list(active_list)
        
        # Pass 2: Try to swap for remaining unselected elements
        # If we couldn't add x, it means there exists y in active_list such that counts[x+y] == 2.
        # We want to remove a y' that resolves this.
        
        # To avoid O(N^2) swaps, we only try a limited number of swaps per x.
        
        # We re-iterate through unselected (which hasn't changed if no adds happened in Pass 1 for this x)
        # Actually, if Pass 1 added elements, unselected changed.
        # So we do this in a single loop or separate loops.
        
        if not progress:
            # No simple adds. Try swaps.
            for x in unselected:
                if _elapsed() > _DEADLINE - 1.0:
                    break
                
                # Check if x is still blocked
                if _can_add(counts, x, in_set):
                    # It became unblocked due to previous swaps in this pass?
                    # Unlikely if we didn't add anything, but possible if we swapped.
                    _add_element(counts, in_set, x, active_list)
                    unselected.remove(x)
                    best_size = len(active_list)
                    best_set = list(active_list)
                    continue
                
                # Find conflicts
                # Conflicts are sums s = x + y where y in active_list and counts[s] == 2
                # We need to remove an element from the set that is part of a pair summing to s.
                
                conflict_y_candidates = []
                for y in active_list:
                    s = x + y
                    if counts[s] == 2:
                        # y is part of a "full" sum with x.
                        # To break this, we must remove y OR remove the other element z such that z + w = s and {z,w} != {x,y} (x not in set).
                        # Wait, the pairs summing to s are among active_list.
                        # Let P_s be the set of pairs (a,b) in active_list such that a+b=s.
                        # Since counts[s]==2, there are 2 such pairs.
                        # If we add x, we add pair (x,y).
                        # So we need to remove an element from one of the existing pairs in P_s.
                        
                        # We can just try removing y itself.
                        conflict_y_candidates.append(y)
                        
                        # Also, we should find the other element z in the pairs for s.
                        # For efficiency, we can just try removing y.
                        # If removing y doesn't work, we might need to remove the partner of y in the other pair?
                        # Actually, if counts[s]==2, there are 2 pairs.
                        # Adding x creates a 3rd pair.
                        # We must remove an element from one of the 2 existing pairs.
                        # The pairs are disjoint? Not necessarily.
                        # Example: s=10. Pairs (2,8) and (4,6).
                        # If x=1, y=9? No, y is in active_list.
                        # If x=1, and 9 is in active_list, s=10.
                        # If counts[10]==2, there are 2 pairs summing to 10 in active_list.
                        # Adding 1 and 9 creates (1,9).
                        # We must remove an element from one of the existing pairs.
                        
                        # Let's find all elements involved in sums with x that are saturated.
                        # We already added y to conflict_y_candidates.
                        # We should also identify the "other" elements in the saturated sums.
                        
                        # To keep it simple, let's just try removing y.
                        # If that fails, we might need a smarter search.
                        # But for now, let's try removing y.
                
                # Randomize the order of removal candidates
                random.shuffle(conflict_y_candidates)
                
                for y_to_remove in conflict_y_candidates:
                    if _elapsed() > _DEADLINE - 1.0:
                        break
                    
                    # Try removing y_to_remove
                    # Check if removing y_to_remove allows x to be added.
                    # We need to simulate the removal.
                    
                    # Instead of full simulate, we can check:
                    # After removing y_to_remove, will counts[x+z] < 2 for all z in (active_list \ {y_to_remove})?
                    # And counts[2*x] < 2?
                    
                    # This is still O(N).
                    # Let's do a quick check.
                    
                    valid_swap = True
                    # The only sums that change are those involving y_to_remove.
                    # But x is new, so we check x + z for all z in active_list \ {y_to_remove}.
                    # Note: counts[x+z] might be 2 currently.
                    # If y_to_remove is not z, the count doesn't change due to removal of y_to_remove.
                    # So if counts[x+z] == 2 and y_to_remove != z, it's still 2.
                    # So we only need to check if removing y_to_remove fixes the specific conflicts caused by x.
                    
                    # Conflicts were sums s = x + y where counts[s] == 2.
                    # If we remove y_to_remove, does counts[s] drop below 2 for all such s?
                    # Only if y_to_remove is part of a pair summing to s.
                    
                    # Let's just simulate the removal and check _can_add.
                    # It's O(N) but N=1000, so 1000 ops.
                    # If we do this for 1000 x's and 1000 y's, it's 10^6 ops per pass.
                    # Passes might be many.
                    
                    # Optimization: Only try a few y_to_remove.
                    # Let's limit to 5 candidates per x.
                    if len(conflict_y_candidates) > 5:
                        conflict_y_candidates = conflict_y_candidates[:5]
                    
                    # Simulate
                    _remove_element(counts, in_set, y_to_remove, active_list)
                    
                    if _can_add(counts, x, in_set):
                        _add_element(counts, in_set, x, active_list)
                        unselected.remove(x)
                        best_size = len(active_list)
                        best_set = list(active_list)
                        progress = True
                        break
                    else:
                        _add_element(counts, in_set, y_to_remove, active_list)
                    
                    if progress:
                        break
                        
        # If no progress in a full cycle, we might be stuck.
        # We could try random restarts, but we only have one construct() call.
        # We can try to "kick" the search by randomly swapping two elements in the set?
        # Or just continue trying. The randomization in unselected and conflict_y_candidates helps.
        
        # To avoid infinite loop of no progress, we can add a small random perturbation.
        # E.g., remove a random element and try to add a random unselected one.
        if not progress and len(active_list) > 10:
            # Random swap attempt
            if random.random() < 0.1:
                y_rand = random.choice(active_list)
                x_rand = random.choice(unselected)
                
                # Try swap
                _remove_element(counts, in_set, y_rand, active_list)
                if _can_add(counts, x_rand, in_set):
                    _add_element(counts, in_set, x_rand, active_list)
                    unselected.remove(x_rand)
                    best_size = len(active_list)
                    best_set = list(active_list)
                else:
                    _add_element(counts, in_set, y_rand, active_list)

    return best_set
