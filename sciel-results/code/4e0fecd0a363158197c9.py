import time
import random

def construct():
    deadline = time.time() + 110  # 10s margin
    initial_elements = [1, 2, 5, 11]
    
    def get_sums(s):
        return {x + y for i, x in enumerate(s) for y in s[i:]}

    def is_valid_set(s):
        # Check if all pairwise sums are distinct
        sums = []
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.append(s[i] + s[j])
        return len(sums) == len(set(sums))

    def try_add(s, sums, x):
        # Returns (new_sums, valid)
        new_sums = set(sums)
        for y in s:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        if len(new_sums) == len(sums) + len(s) + 1:
            return new_sums, True
        return sums, False

    # 1. Greedy Initialization with Randomization
    # We will run several randomized greedy passes to find a good starting set
    best_set = list(initial_elements)
    
    for _ in range(50): # 50 restarts of greedy
        if time.time() > deadline - 5:
            break
            
        current_set = list(initial_elements)
        current_sums = get_sums(current_set)
        
        # Randomized Greedy
        candidates = list(range(12, 285))
        random.shuffle(candidates)
        
        for x in candidates:
            if x in current_set:
                continue
            new_sums, valid = try_add(current_set, current_sums, x)
            if valid:
                current_set.append(x)
                current_sums = new_sums
                if len(current_set) > len(best_set):
                    best_set = current_set
                    if len(best_set) == 20:
                        return best_set

    # 2. Local Search to improve best_set
    # We try to add more elements to best_set
    # If we can't add, we swap to explore
    
    current_set = list(best_set)
    current_sums = get_sums(current_set)
    
    # We want to reach size 20.
    # If current size is S, we try to add.
    # If stuck, we swap.
    
    # Optimization: Precompute valid candidates for the current set?
    # It changes with every swap.
    
    # Time budget for local search
    local_search_deadline = deadline - 5
    
    # We will iterate, trying to add. If no add possible, do a random swap.
    # Keep track of best size found.
    
    best_size_found = len(current_set)
    best_set_found = list(current_set)
    
    # To make local search effective, we need to move through the space of size-k sets
    # to find one that can be extended to k+1.
    
    # Let's implement a simple hill climbing with swaps.
    # State: current_set
    # Move: Swap x in current_set with y not in current_set.
    # Accept if valid.
    # Goal: Find a state where an addition is possible.
    
    # We can also just try to add directly first.
    
    while time.time() < local_search_deadline:
        # Try to add
        added = False
        # Randomly shuffle candidates to avoid bias
        candidates = [x for x in range(1, 285) if x not in current_set]
        random.shuffle(candidates)
        
        for x in candidates:
            new_sums, valid = try_add(current_set, current_sums, x)
            if valid:
                current_set.append(x)
                current_sums = new_sums
                added = True
                if len(current_set) > best_size_found:
                    best_size_found = len(current_set)
                    best_set_found = list(current_set)
                    if best_size_found == 20:
                        return best_set_found
                break
        
        if not added:
            # Perform a swap to explore
            if len(current_set) > len(initial_elements): # Don't swap prescribed elements
                # Pick a random element to swap (not prescribed)
                swappable_indices = [i for i, x in enumerate(current_set) if x not in initial_elements]
                if swappable_indices:
                    idx = random.choice(swappable_indices)
                    x_to_remove = current_set[idx]
                    
                    # Pick a random candidate to add
                    candidates_swap = [x for x in range(1, 285) if x not in current_set]
                    random.shuffle(candidates_swap)
                    
                    swap_made = False
                    for y in candidates_swap:
                        # Try swap
                        new_set = current_set.copy()
                        new_set[idx] = y
                        # Check validity of new_set
                        # Efficient check:
                        # Remove sums involving x_to_remove
                        # Add sums involving y
                        
                        # For simplicity and speed, just check full validity if set is small (<20)
                        # Or implement efficient delta check.
                        # Delta check:
                        # Sums to remove: {x_to_remove + z for z in current_set if z != x_to_remove} U {2*x_to_remove}
                        # Sums to add: {y + z for z in current_set if z != x_to_remove} U {2*y}
                        
                        # Let's just use the full check for now as N is small
                        if is_valid_set(new_set):
                            current_set = new_set
                            current_sums = get_sums(current_set)
                            swap_made = True
                            break
                    # If no swap found, we are stuck in a local optimum for this configuration
                    # We might need a larger jump (restart)
                    if not swap_made and random.random() < 0.1:
                         # Restart from best found or initial
                         current_set = list(best_set_found)
                         current_sums = get_sums(current_set)

    return best_set_found
