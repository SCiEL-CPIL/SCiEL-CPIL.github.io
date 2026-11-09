"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from math import gcd
from functools import reduce

def _lcm(a, b):
    return a // gcd(a, b) * b

def _lcm_list(lst):
    return reduce(_lcm, lst)

def construct():
    """
    Return a nonempty subset of [1,22] such that ALL reciprocal subset sums,
    including the empty sum, are distinct. Maximize cardinality.
    
    Strategy: 
    1. Convert problem to integer weights w_i = LCM(1..22) / i.
    2. Find largest subset S of indices such that subset sums of {w_i : i in S} are distinct.
    3. Use local search:
       - State: Current set S and its subset sums.
       - Move: Add, Swap, or Remove.
       - Optimization: For 'Add', check collision against existing sums in O(2^|S|).
       - For 'Swap', recompute sums.
    """
    N = 22
    universe = list(range(1, N + 1))
    
    # Precompute LCM of 1..22
    L = _lcm_list(universe)
    
    # Integer weights: w_i = L // i
    weights = {i: L // i for i in universe}
    
    # Helper: Compute all subset sums for a list of indices
    def get_subset_sums(indices):
        sums = {0}
        for idx in indices:
            w = weights[idx]
            new_sums = set()
            for s in sums:
                new_sums.add(s + w)
            sums.update(new_sums)
        return sums

    # Helper: Check if adding index 'idx' to current sums causes a collision
    # Returns True if valid (no collision)
    def can_add(idx, current_sums):
        w = weights[idx]
        # Collision if any s in current_sums satisfies s + w in current_sums
        max_sum = max(current_sums)
        if w > max_sum:
            return True
        
        for s in current_sums:
            if s + w in current_sums:
                return False
        return True

    # Initial greedy construction
    best_set = []
    best_size = 0
    
    # Try multiple random greedy starts
    start_time = time.time()
    timeout = 55.0
    
    def greedy_start():
        S = []
        sums = {0}
        # Sort by weight descending (i.e., index ascending)
        for i in sorted(universe, key=lambda x: weights[x], reverse=True):
            if can_add(i, sums):
                S.append(i)
                # Update sums
                w = weights[i]
                new_sums = set()
                for s in sums:
                    new_sums.add(s + w)
                sums.update(new_sums)
        return S, sums

    # Try a few greedy starts
    for _ in range(5):
        S, sums = greedy_start()
        if len(S) > best_size:
            best_size = len(S)
            best_set = S[:]
            
    # Now local search on the best set found
    current_set = best_set[:]
    current_sums = get_subset_sums(current_set)
    current_size = len(current_set)
    
    # If best_set is empty, start from scratch
    if not current_set:
        current_set = [1] 
        current_sums = get_subset_sums(current_set)
        current_size = 1
        best_size = 1
        best_set = [1]

    stall_counter = 0
    
    while time.time() - start_time < timeout:
        # 1. Try to add
        added = False
        candidates_to_add = [i for i in universe if i not in current_set]
        if candidates_to_add:
            # Shuffle to randomize
            random.shuffle(candidates_to_add)
            for i in candidates_to_add:
                if can_add(i, current_sums):
                    # Accept
                    current_set.append(i)
                    w = weights[i]
                    # Update sums
                    new_sums = set()
                    for s in current_sums:
                        new_sums.add(s + w)
                    current_sums.update(new_sums)
                    current_size += 1
                    if current_size > best_size:
                        best_size = current_size
                        best_set = current_set[:]
                    stall_counter = 0
                    added = True
                    break
        
        if added:
            continue
            
        # 2. No element could be added. Try to swap.
        swapped = False
        swap_attempts = 10
        for _ in range(swap_attempts):
            if len(current_set) <= 1:
                break
            i_remove = random.choice(current_set)
            candidates_to_add = [i for i in universe if i not in current_set]
            if not candidates_to_add:
                break
            i_add = random.choice(candidates_to_add)
            
            # Construct new set
            new_set = [x for x in current_set if x != i_remove]
            new_set.append(i_add)
            
            # Check validity
            S_base = [x for x in new_set if x != i_add]
            if len(S_base) > 0:
                sums_base = get_subset_sums(S_base)
            else:
                sums_base = {0}
                
            if can_add(i_add, sums_base):
                # Valid swap
                current_set = new_set
                current_sums = get_subset_sums(current_set) # Update full sums
                # Size unchanged
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set[:]
                stall_counter += 1
                swapped = True
                break
        
        if swapped:
            continue
            
        # 3. No swap worked. Remove an element to explore.
        if len(current_set) > 1:
            # Remove a random element
            i_remove = random.choice(current_set)
            current_set = [x for x in current_set if x != i_remove]
            current_sums = get_subset_sums(current_set)
            current_size -= 1
            stall_counter += 1
        else:
            # Stuck at size 1?
            if stall_counter > 100:
                # Restart
                S, sums = greedy_start()
                current_set = S
                current_sums = sums
                current_size = len(S)
                stall_counter = 0
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set[:]
            else:
                # Just wait / try random add again?
                # Force restart if stuck
                S, sums = greedy_start()
                current_set = S
                current_sums = sums
                current_size = len(S)
                stall_counter = 0
                if current_size > best_size:
                    best_size = current_size
                    best_set = current_set[:]

    return best_set
# EVOLVE-BLOCK-END
