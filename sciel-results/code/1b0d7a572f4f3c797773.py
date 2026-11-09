import time
import random
from itertools import combinations

# LCM of 1..22
_LCM = 232792560

def _is_valid_fast(subset):
    """
    Fast check if all subset sums of reciprocals are distinct.
    Uses incremental collision detection.
    """
    if not subset:
        return True
    
    # Convert to integer weights, sort in descending order for better early termination
    weights = sorted([_LCM // i for i in subset], reverse=True)
    k = len(weights)
    
    # Generate all subset sums incrementally, check for collisions
    sums = {0}
    for w in weights:
        # Add w to all existing sums
        new_sums = set()
        for s in sums:
            ns = s + w
            if ns in sums:
                return False
            new_sums.add(ns)
        sums |= new_sums
        
    return True

def _try_add_element(current, candidate, weights_cache):
    """
    Try to add candidate to current set.
    Returns True if valid, False otherwise.
    """
    if candidate in current:
        return False
    
    # Quick check: if candidate's weight is too small relative to existing sums,
    # it might cause collisions. But we need to check properly.
    return _is_valid_fast(list(current) + [candidate])

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin (90s limit, leave buffer)
    
    # Precompute weights for all numbers 1..22
    all_weights = {i: _LCM // i for i in range(1, 23)}
    
    # Start with a known good seed
    best_subset = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_subset)
    
    # Strategy 1: Greedy construction with multiple random orderings
    # Try different heuristics for ordering
    for iteration in range(3000):
        if time.time() > deadline:
            break
        
        # Different strategies for ordering
        if iteration % 3 == 0:
            # Random order
            candidates = list(range(1, 23))
            random.shuffle(candidates)
        elif iteration % 3 == 1:
            # Prefer larger numbers (smaller weights) first - less likely to cause collisions
            candidates = list(range(22, 0, -1))
            random.shuffle(candidates)  # Add some randomness
        else:
            # Prefer smaller numbers (larger weights) first
            candidates = list(range(1, 23))
            random.shuffle(candidates)
        
        current = []
        current_set = set()
        
        for c in candidates:
            if c in current_set:
                continue
            # Try to add
            if _is_valid_fast(current + [c]):
                current.append(c)
                current_set.add(c)
        
        if len(current) > best_size:
            best_size = len(current)
            best_subset = current
            # If we found a very large set, we can be more aggressive
            if best_size >= 17:
                # Try to extend even more
                remaining = [x for x in range(1, 23) if x not in current_set]
                for c in remaining:
                    if time.time() > deadline:
                        break
                    if _is_valid_fast(current + [c]):
                        current.append(c)
                        current_set.add(c)
                if len(current) > best_size:
                    best_size = len(current)
                    best_subset = current

    # Strategy 2: Local search - try to improve best_subset
    if time.time() < deadline - 5:
        # Try swapping elements in best_subset
        best_set = set(best_subset)
        for _ in range(500):
            if time.time() > deadline:
                break
            
            # Try to add an element not in best_set
            added = False
            for c in range(1, 23):
                if c not in best_set:
                    if _is_valid_fast(list(best_set) + [c]):
                        best_set.add(c)
                        added = True
                        break
            
            if not added:
                # Try swapping: remove one, add another
                for remove_c in list(best_set):
                    if time.time() > deadline:
                        break
                    temp_set = set(best_set)
                    temp_set.remove(remove_c)
                    for add_c in range(1, 23):
                        if add_c not in temp_set:
                            if _is_valid_fast(list(temp_set) + [add_c]):
                                temp_set.add(add_c)
                                if len(temp_set) > len(best_set):
                                    best_set = temp_set
                                break
    
        if len(best_set) > best_size:
            best_size = len(best_set)
            best_subset = list(best_set)

    # Strategy 3: Try specific known good structures
    # Sets with large gaps tend to work well
    if time.time() < deadline - 2:
        # Try powers of 2 like sets
        test_sets = [
            [1, 2, 4, 8, 16],  # Powers of 2
            [1, 2, 4, 8, 16, 22],
            [1, 2, 4, 8, 16, 20, 22],
            [1, 2, 4, 8, 16, 21, 22],
            [1, 2, 4, 8, 16, 19, 22],
            [1, 2, 4, 8, 16, 18, 22],
            [1, 2, 4, 8, 16, 17, 22],
            [1, 2, 4, 8, 16, 15, 22],
            [1, 2, 4, 8, 16, 14, 22],
            [1, 2, 4, 8, 16, 13, 22],
        ]
        
        for ts in test_sets:
            if _is_valid_fast(ts):
                if len(ts) > best_size:
                    best_size = len(ts)
                    best_subset = ts
                # Try to extend
                ts_set = set(ts)
                for c in range(1, 23):
                    if c not in ts_set:
                        if _is_valid_fast(list(ts_set) + [c]):
                            ts_set.add(c)
                if len(ts_set) > best_size:
                    best_size = len(ts_set)
                    best_subset = list(ts_set)

    return best_subset
