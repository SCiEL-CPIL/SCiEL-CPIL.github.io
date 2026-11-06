import time
import numpy as np
from itertools import combinations
import random
import math

# EVOLVE-BLOCK-START
# Precompute weights based on LCM of 1..22
_LCM_22 = 1
for _i in range(1, 23):
    _LCM_22 = math.lcm(_LCM_22, _i)

_WEIGHTS = {}
for _i in range(1, 23):
    _WEIGHTS[_i] = _LCM_22 // _i

def _can_add(sums, seen, w):
    """
    Check if adding weight w to the set of sums 'sums' creates any duplicates.
    Returns True if valid, and the new list of sums if valid.
    """
    new_sums = [s + w for s in sums]
    for ns in new_sums:
        if ns in seen:
            return False, None
    return True, new_sums

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 55s safety margin for 60s limit
    
    best_set = []
    best_size = 0
    
    # We will use randomized local search with incremental sum updates.
    # State: current_set (set of ints), current_sums (list of int sums), seen_sums (set of int sums)
    
    num_restarts = 0
    
    while time.time() < deadline:
        num_restarts += 1
        
        # Randomize the order of elements to try
        elements = list(range(1, 23))
        random.shuffle(elements)
        
        current_set = set()
        current_sums = [0]
        seen_sums = {0}
        
        # Greedy phase: try to add elements from the randomized list
        for e in elements:
            if e in current_set:
                continue
            w = _WEIGHTS[e]
            valid, new_sums = _can_add(current_sums, seen_sums, w)
            if valid:
                current_set.add(e)
                current_sums.extend(new_sums)
                seen_sums.update(new_sums)
                
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            
        # Local search: try to improve current_set by swaps or additions
        # We try to add any element not in the set. If none, try swapping one out and one in.
        # To avoid getting stuck, we do a limited number of swap attempts.
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # 1. Try to add any element
            added = False
            for e in range(1, 23):
                if e in current_set:
                    continue
                w = _WEIGHTS[e]
                valid, new_sums = _can_add(current_sums, seen_sums, w)
                if valid:
                    current_set.add(e)
                    current_sums.extend(new_sums)
                    seen_sums.update(new_sums)
                    improved = True
                    added = True
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = list(current_set)
                    break # Restart the loop to try adding more
            
            if added:
                continue # Restart while loop to try adding more
            
            # 2. If no addition possible, try swaps
            # Pick a random element to remove, then try to add a different one
            # To be efficient, we only do a few random swap attempts per restart
            for _ in range(50): # Limit swap attempts to keep restarts fast
                if time.time() > deadline:
                    break
                
                if not current_set:
                    break
                    
                remove_e = random.choice(list(current_set))
                
                # Remove remove_e from state
                # This requires recalculating sums without remove_e, which is expensive if done naively.
                # However, we can reconstruct sums from scratch for the remaining set.
                # Since size is small (<15), this is acceptable.
                
                temp_set = current_set - {remove_e}
                
                # Rebuild sums for temp_set
                temp_sums = [0]
                temp_seen = {0}
                for e in temp_set:
                    w = _WEIGHTS[e]
                    valid, new_sums = _can_add(temp_sums, temp_seen, w)
                    # It must be valid because it was a subset of a valid set
                    temp_sums.extend(new_sums)
                    temp_seen.update(new_sums)
                
                # Try to add a different element to temp_set
                swapped = False
                for add_e in range(1, 23):
                    if add_e in temp_set:
                        continue
                    w = _WEIGHTS[add_e]
                    valid, new_sums = _can_add(temp_sums, temp_seen, w)
                    if valid:
                        # Perform the swap
                        current_set = temp_set
                        current_set.add(add_e)
                        current_sums = temp_sums
                        current_sums.extend(new_sums)
                        seen_sums = temp_seen
                        seen_sums.update(new_sums)
                        
                        improved = True
                        swapped = True
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = list(current_set)
                        break # Restart the while loop to try adding more
                
                if swapped:
                    break # Break out of swap attempts, go back to while loop

    if not best_set:
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
        
    return best_set
# EVOLVE-BLOCK-END
