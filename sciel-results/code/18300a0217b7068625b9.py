import time
import numpy as np
import random
from itertools import combinations

# EVOLVE-BLOCK-START
def _is_dissociated_subset(S, recips=None):
    """
    Checks if a set of integers S is dissociated in the reciprocals.
    i.e., all subset sums of {1/s for s in S} are distinct.
    """
    if not S:
        return True
    
    n = len(S)
    
    if recips is None:
        reciprocals = [1.0 / s for s in S]
    else:
        reciprocals = [recips[s] for s in S]
    
    # Generate all subset sums
    sums = [0.0]
    for r in reciprocals:
        # Add r to all existing sums
        new_sums = [s + r for s in sums]
        sums.extend(new_sums)
        
    # Check for duplicates
    sorted_sums = sorted(sums)
    for i in range(len(sorted_sums) - 1):
        if sorted_sums[i] == sorted_sums[i+1]:
            return False
            
    return True

def _greedy_build(order, recips):
    """Greedily build a dissociated set following the given order."""
    current_set = []
    used = set()
    
    for x in order:
        if x in used:
            continue
        new_set = current_set + [x]
        if _is_dissociated_subset(new_set, recips):
            current_set = new_set
            used.add(x)
    
    return current_set

def _local_search_add(best_set, recips, deadline):
    """Try to add elements to best_set to make it larger."""
    current = list(best_set)
    current_set = set(current)
    changed = True
    
    while changed and time.time() < deadline:
        changed = False
        for x in range(1, 23):
            if x not in current_set:
                candidate = current + [x]
                if _is_dissociated_subset(candidate, recips):
                    current = candidate
                    current_set.add(x)
                    changed = True
                    break
    
    return current

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit (conservative)
    
    # Precompute reciprocals
    recips = {i: 1.0/i for i in range(1, 23)}
    universe = list(range(1, 23))
    
    # Known good seed
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    
    # Strategy 1: Greedy with various orders
    orders = [
        list(range(1, 23)),
        list(range(22, 0, -1)),
    ]
    
    # Add some random permutations
    random.seed(42)
    np.random.seed(42)
    
    for _ in range(100):
        if time.time() > deadline - 15:
            break
        cand_order = list(range(1, 23))
        random.shuffle(cand_order)
        orders.append(cand_order)
    
    for order in orders:
        if time.time() > deadline - 15:
            break
            
        current_set = _greedy_build(order, recips)
        
        if len(current_set) > best_size:
            best_set = current_set
            best_size = len(current_set)
            
        # Early exit if we reached a very high number
        if best_size >= 18:
            break

    # Strategy 2: Local search refinement
    if time.time() < deadline - 10:
        refined = _local_search_add(best_set, recips, deadline - 5)
        if len(refined) > best_size:
            best_set = refined
            best_size = len(best_set)

    # Strategy 3: Random sampling for larger sets
    max_target = min(22, best_size + 4)
    
    for target_size in range(max_target, best_size, -1):
        if time.time() > deadline - 5:
            break
        
        attempts = 0
        max_attempts = 30000 if target_size <= 14 else 5000
        
        while attempts < max_attempts:
            if time.time() > deadline - 2:
                break
            
            candidate = random.sample(universe, target_size)
            if _is_dissociated_subset(candidate, recips):
                best_set = candidate
                best_size = target_size
                # Do local search on this new best
                if time.time() < deadline - 2:
                    refined = _local_search_add(best_set, recips, deadline - 1)
                    if len(refined) > best_size:
                        best_set = refined
                        best_size = len(best_set)
                break
            attempts += 1

    # Final polish
    if time.time() < deadline - 1 and best_size < 20:
        refined = _local_search_add(best_set, recips, deadline - 0.5)
        if len(refined) > best_size:
            best_set = refined
            best_size = len(best_set)
        
    # Fallback
    if not best_set:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
# EVOLVE-BLOCK-END
