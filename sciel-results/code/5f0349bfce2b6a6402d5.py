"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s limit

    # Prescribed elements
    prescribed = [1, 2, 5, 11]
    N = 284
    prescribed_set = set(prescribed)

    def init_sums(s):
        sums = set()
        for i, x in enumerate(s):
            for y in s[i:]:
                sums.add(x + y)
        return sums

    def can_add(s, sums, x):
        """Check if x can be added to Sidon set s with existing sums."""
        for y in s:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    def add_element(s, sums, x):
        """Add x to s and update sums. Assumes can_add is true."""
        s.append(x)
        for y in s[:-1]:
            sums.add(x + y)
        sums.add(2 * x)

    # Strategy 1: Simple greedy baseline
    s1 = list(prescribed)
    sums1 = init_sums(s1)
    for x in range(12, N + 1):
        if can_add(s1, sums1, x):
            add_element(s1, sums1, x)

    best_set = s1[:]
    best_size = len(best_set)

    # Strategy 2: Randomized greedy restarts
    random.seed(42)
    
    for restart in range(500):
        if time.time() > DEADLINE:
            break

        candidates = list(range(12, N + 1))
        random.shuffle(candidates)

        s, sums = list(prescribed), init_sums(prescribed)
        for x in candidates:
            if can_add(s, sums, x):
                add_element(s, sums, x)

        if len(s) > best_size:
            best_set = s[:]
            best_size = len(best_set)
            if best_size == 20:
                return best_set

    # Strategy 3: Local Search (k-opt)
    # We try to remove k elements and add k different elements to improve the set.
    # This helps escape local maxima of size 19.
    
    def local_search(current_best, current_size):
        best_set = current_best[:]
        best_size = current_size
        s = best_set[:]
        sums = init_sums(s)
        
        # Precompute list of all non-prescribed candidates
        all_candidates = [x for x in range(12, N + 1) if x not in prescribed_set]
        
        while time.time() < DEADLINE:
            # Choose k randomly from 1 to 3
            k = random.choice([1, 2, 3])
            
            # Get current non-prescribed elements
            current_non_prescribed = [x for x in s if x not in prescribed_set]
            
            if len(current_non_prescribed) < k:
                break
            
            # Randomly select k elements to remove
            to_remove = random.sample(current_non_prescribed, k)
            
            # Create a trial set
            s_trial = [x for x in s if x not in to_remove]
            sums_trial = init_sums(s_trial)
            
            # Identify available candidates to add
            # We want to add k elements that are not in s_trial
            available = [x for x in all_candidates if x not in s_trial]
            
            if len(available) < k:
                continue
                
            # Shuffle available to get a random subset to try
            random.shuffle(available)
            
            # Try to add k elements from available to s_trial
            # We don't need to add exactly k, we just want to maximize size.
            # But since we removed k, we aim to add at least k to restore size, 
            # or more if possible (though max is 20).
            
            added_count = 0
            for x in available:
                if added_count >= k:
                    break
                if can_add(s_trial, sums_trial, x):
                    add_element(s_trial, sums_trial, x)
                    added_count += 1
            
            # Check if the trial set is better
            if len(s_trial) > best_size:
                best_set = s_trial[:]
                best_size = len(best_set)
                s = s_trial[:]
                sums = sums_trial
                if best_size == 20:
                    return best_set
            elif len(s_trial) == best_size:
                # Even if size is same, we might have found a different configuration
                # that allows for future improvements. We can accept it with some probability
                # or just continue. For now, let's just keep the best size found.
                pass

        return best_set, best_size

    best_set, best_size = local_search(best_set, best_size)

    # Strategy 4: Backtracking for size 20 specifically
    # If we haven't found size 20, we try a directed backtracking.
    if best_size < 20:
        target_size = 20
        # We need to pick target_size - len(prescribed) = 16 elements
        # from the remaining 273 candidates.
        
        # Heuristic: Sort candidates by "potential"
        # A candidate x is "good" if it conflicts with few existing sums in a potential optimal set.
        # Since we don't know the optimal set, we use the current best set as a guide.
        
        # Let's try to build the set from scratch using backtracking with MRV
        # but guided by the greedy result.
        
        def backtrack_search(current_set, current_sums, remaining_candidates, target_size):
            if len(current_set) == target_size:
                return current_set[:]
            
            if not remaining_candidates:
                return None
            
            # Pruning: if we can't reach target size
            if len(current_set) + len(remaining_candidates) < target_size:
                return None
                
            # Heuristic: Try candidates that are "compatible" with the current set
            # We iterate through remaining candidates.
            # To speed up, we can shuffle them or order them by value.
            
            # Try each candidate
            for i in range(len(remaining_candidates)):
                x = remaining_candidates[i]
                
                if can_add(current_set, current_sums, x):
                    # Add x
                    new_set = current_set + [x]
                    new_sums = set(current_sums)
                    for y in current_set:
                        new_sums.add(x + y)
                    new_sums.add(2 * x)
                    
                    # Recurse
                    result = backtrack_search(new_set, new_sums, remaining_candidates[i+1:], target_size)
                    if result:
                        return result
            
            return None

        # The above backtracking is O(N^K) which is too slow for K=16.
        # We need a smarter approach.
        # Let's try to "repair" the best_set found in Strategy 3.
        # We know best_size is likely 19.
        # We try to swap 1 element for 2, or 2 for 3, etc., specifically aiming for +1 size.
        
        s = best_set[:]
        sums = init_sums(s)
        all_candidates = [x for x in range(12, N + 1) if x not in s]
        
        for _ in range(1000):
            if time.time() > DEADLINE:
                break
            
            # Try to remove 1 element and add 2
            if len(s) > len(prescribed):
                removable = [x for x in s if x not in prescribed_set]
                if removable:
                    x_remove = random.choice(removable)
                    s_trial = [x for x in s if x != x_remove]
                    sums_trial = init_sums(s_trial)
                    
                    # Try to add 2 elements
                    candidates = [x for x in all_candidates if x not in s_trial]
                    random.shuffle(candidates)
                    
                    added = 0
                    for x_add in candidates:
                        if added == 2:
                            break
                        if can_add(s_trial, sums_trial, x_add):
                            add_element(s_trial, sums_trial, x_add)
                            added += 1
                            
                    if len(s_trial) > best_size:
                        best_set = s_trial[:]
                        best_size = len(best_set)
                        s = s_trial[:]
                        sums = sums_trial
                        if best_size == 20:
                            return best_set

    return best_set

# EVOLVE-BLOCK-END
