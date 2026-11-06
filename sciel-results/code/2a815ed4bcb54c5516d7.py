import random
import time
import numpy as np

def construct():
    n = 512
    
    # Step 1: Base-3 digit-based construction
    # Select numbers in [1, n] whose base-3 representation contains no digit '2'
    base_set = set()
    for x in range(1, n + 1):
        num = x
        valid = True
        while num > 0:
            if num % 3 == 2:
                valid = False
                break
            num //= 3
        if valid:
            base_set.add(x)
            
    # Represent set as a boolean array for fast lookups
    # in_set[i] is True if i is in the set
    in_set = np.zeros(n + 1, dtype=bool)
    for x in base_set:
        in_set[x] = True
        
    def can_add(x, in_set_arr):
        # Check if adding x creates a 4-AP
        # 1. x is 1st term: x, x+d, x+2d, x+3d
        max_d1 = (n - x) // 3
        for d in range(1, max_d1 + 1):
            if in_set_arr[x + d] and in_set_arr[x + 2*d] and in_set_arr[x + 3*d]:
                return False
        # 2. x is 2nd term: x-d, x, x+d, x+2d
        max_d2 = min(x - 1, (n - x) // 2)
        for d in range(1, max_d2 + 1):
            if in_set_arr[x - d] and in_set_arr[x + d] and in_set_arr[x + 2*d]:
                return False
        # 3. x is 3rd term: x-2d, x-d, x, x+d
        max_d3 = min((x - 1) // 2, n - x)
        for d in range(1, max_d3 + 1):
            if in_set_arr[x - 2*d] and in_set_arr[x - d] and in_set_arr[x + d]:
                return False
        # 4. x is 4th term: x-3d, x-2d, x-d, x
        max_d4 = (x - 1) // 3
        for d in range(1, max_d4 + 1):
            if in_set_arr[x - 3*d] and in_set_arr[x - 2*d] and in_set_arr[x - d]:
                return False
        return True

    def greedy_add(in_set_arr, order):
        added_count = 0
        for x in order:
            if not in_set_arr[x]:
                if can_add(x, in_set_arr):
                    in_set_arr[x] = True
                    added_count += 1
        return added_count

    # Initial Greedy Pass
    # Try multiple random orders to find a good initial packing
    best_in_set = in_set.copy()
    best_size = np.sum(best_in_set)
    
    start_time = time.time()
    time_limit = 55.0
    
    # 1. Multiple Random Greedy Restarts
    # This helps escape bad local minima from the base set
    for _ in range(20):
        if time.time() - start_time > time_limit:
            break
        trial_in_set = in_set.copy()
        order = list(range(1, n + 1))
        random.shuffle(order)
        greedy_add(trial_in_set, order)
        size = np.sum(trial_in_set)
        if size > best_size:
            best_size = size
            best_in_set = trial_in_set.copy()
            
    # 2. Local Search: Remove-Add
    # Try removing each element and then greedily adding back
    # We keep a list of "candidates" (elements not in set) sorted by some heuristic?
    # Just use random order for adding.
    
    improved = True
    while improved and time.time() - start_time < time_limit:
        improved = False
        current_elements = np.nonzero(best_in_set)[0]
        
        # Shuffle elements to try in random order
        random.shuffle(current_elements)
        
        for y in current_elements:
            if time.time() - start_time > time_limit:
                break
            
            # Remove y
            trial_in_set = best_in_set.copy()
            trial_in_set[y] = False
            
            # Try to add elements not in the set
            # To be efficient, we don't re-scan all 1..n.
            # We can just run a greedy pass on the remaining candidates.
            # However, generating the list of candidates and shuffling is O(n).
            # n=512 is small, so it's fine.
            
            candidates = [x for x in range(1, n + 1) if not trial_in_set[x]]
            random.shuffle(candidates)
            
            # Greedy add from candidates
            for x in candidates:
                if can_add(x, trial_in_set):
                    trial_in_set[x] = True
            
            new_size = np.sum(trial_in_set)
            
            if new_size > best_size:
                best_size = new_size
                best_in_set = trial_in_set
                improved = True
                # Restart the loop over elements with the new set
                break
                
    result = [int(x) for x in np.nonzero(best_in_set)[0]]
    return sorted(result)
