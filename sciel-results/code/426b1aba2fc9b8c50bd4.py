"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time

def construct():
    def get_subset_sums(elements):
        """Returns a set of all non-empty subset sums."""
        sums = {0}
        for e in elements:
            new_sums = {s + e for s in sums}
            sums |= new_sums
        sums.discard(0)
        return sums

    def is_valid(elements):
        """
        Checks if the set of elements satisfies:
        1. All non-empty subset sums are distinct.
        2. No two distinct subset sums divide each other.
        """
        if not elements:
            return True
        
        sums = get_subset_sums(elements)
        
        # Check distinctness: The number of sums must be 2^len(elements) - 1
        if len(sums) != (1 << len(elements)) - 1:
            return False
        
        # Check divisibility
        # For each sum s, check if any other sum t is a multiple of s (t = k*s, k>=2)
        # Or if s is a multiple of t.
        # Since we just need to check if ANY pair divides, we can iterate.
        # To be efficient, we can check for each s if there exists k>=2 such that k*s in sums.
        # This covers s | t. If t | s, then s = k*t, so when we iterate t, we find s.
        # So checking s | t for all s is sufficient.
        
        max_sum = max(sums)
        for s in sums:
            # Check multiples of s
            multiple = s * 2
            while multiple <= max_sum:
                if multiple in sums:
                    return False
                multiple += s
                
        return True

    def greedy_randomize(order, time_limit=50):
        """
        Greedily add elements from the given order if they maintain validity.
        """
        selected = []
        start_time = time.time()
        
        for candidate in order:
            if time.time() - start_time > time_limit:
                break
            if candidate in selected:
                continue
            
            new_selected = selected + [candidate]
            if is_valid(new_selected):
                selected = new_selected
                
        return selected

    def local_search_improve(current_set, time_limit=50):
        """
        Try to improve the set by swapping elements or adding new ones.
        This is a simple hill climbing.
        """
        selected = list(current_set)
        start_time = time.time()
        improved = True
        attempts = 0
        
        while improved and (time.time() - start_time) < time_limit:
            improved = False
            attempts += 1
            if attempts > 1000: break # Safety break
            
            # Try adding a random element
            # We want to try elements that might fit.
            # Let's pick a random element from 1..256 not in selected
            available = [i for i in range(1, 257) if i not in selected]
            if available:
                random.shuffle(available)
                # Try a few random additions
                for cand in available[:10]:
                    new_set = selected + [cand]
                    if is_valid(new_set):
                        selected = new_set
                        improved = True
                        break
            
            # Try swapping: remove one, add one
            if len(selected) > 1:
                # Pick a random element to remove
                idx = random.randint(0, len(selected) - 1)
                removed = selected[idx]
                temp_selected = [x for i, x in enumerate(selected) if i != idx]
                
                # Try to add a random new element
                available = [i for i in range(1, 257) if i not in temp_selected]
                if available:
                    random.shuffle(available)
                    for cand in available[:10]:
                        new_set = temp_selected + [cand]
                        if is_valid(new_set):
                            # Only accept if size is same or larger? 
                            # We are swapping, so size is same.
                            # We accept to explore.
                            selected = new_set
                            improved = True
                            break

        return selected

    best_set = []
    start_time = time.time()
    limit = 55.0
    
    # Strategy: Multiple random restarts with different biases
    # 1. Random order
    # 2. Descending order (large numbers first)
    # 3. Ascending order (small numbers first)
    # 4. Prime bias?
    
    for i in range(50): # Max 50 restarts
        if time.time() - start_time > limit:
            break
        
        # Generate a random order
        candidates = list(range(1, 257))
        
        # Mix strategies
        if i % 3 == 0:
            random.shuffle(candidates)
        elif i % 3 == 1:
            candidates.sort(reverse=True)
        else:
            candidates.sort()
            
        current = greedy_randomize(candidates, time_limit=limit - (time.time() - start_time) - 5)
        
        if len(current) > len(best_set):
            best_set = current
            
        # Local search on the best found so far or current?
        # Let's do local search on current if it's good enough
        if len(current) >= len(best_set) and len(current) > 0:
            # Only do local search if we have time
            remaining = limit - (time.time() - start_time)
            if remaining > 5:
                improved = local_search_improve(current, time_limit=min(5, remaining))
                if len(improved) > len(best_set):
                    best_set = improved

    # Final verification
    if not is_valid(best_set):
        # Fallback to a small known valid set
        # [1] is valid. [1, 3] -> sums 1, 3, 4. 1|3, 1|4. Invalid.
        # [2] valid.
        # [3] valid.
        # [1, 5] -> 1, 5, 6. 1|5, 1|6. Invalid.
        # It seems 1 is very dangerous.
        # [2, 5] -> 2, 5, 7. 2 does not divide 5 or 7. 5 does not divide 2 or 7. 7 does not divide 2 or 5. Valid.
        # [2, 5, 11] -> sums: 2, 5, 7, 11, 13, 16, 18.
        # 2|? 2|16, 2|18. Invalid.
        # So even size 3 is hard.
        best_set = [2, 5]

    return [int(x) for x in best_set]

# EVOLVE-BLOCK-END
