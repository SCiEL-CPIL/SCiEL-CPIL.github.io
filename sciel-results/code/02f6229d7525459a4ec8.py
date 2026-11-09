<<<<<<< SEARCH
def _local_search(selected, time_limit):
    """Local search with 1-add and 1-swap moves."""
    start = time.time()
    best = list(selected)
    best_size = len(best)
    best_arr = np.array(best)
    
    while time.time() - start < time_limit:
        improved = False
        
        # 1. Try to add any element
        if len(best) < MAX_SIZE:
            # To speed this up, we can shuffle candidates or just iterate.
            # Iterating 192 candidates * O(S^3) check is expensive if S is large.
            # But vectorized check is fast. Let's try a few random candidates first, then all.
            candidates = [x for x in range(1, N + 1) if x not in best]
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                if time.time() - start > time_limit:
                    break
                if _can_add_vectorized(x, best_arr):
                    best.append(x)
                    best_arr = np.array(best)
                    improved = True
                    added = True
                    break
            
            if added:
                continue # Restart search with new set
        
        # 2. Try 1-swap (remove one, add one)
        # Only do this if we are stuck or trying to improve diversity?
        # Since we want max cardinality, and add failed, swap might not increase size.
        # However, a swap might allow a future add.
        # Given time constraints, we might skip complex swaps if "Add" is the primary driver.
        # But let's try a limited number of swaps to escape local minima.
        
        if len(best) > 0:
            # Try removing a random element and adding a random candidate
            # This is a random walk.
            # Let's try a few random swaps.
            for _ in range(10): # Limit swaps per iteration to avoid timeout
                if time.time() - start > time_limit:
                    break
                i = random.randint(0, len(best) - 1)
                temp_list = best[:i] + best[i+1:]
                temp_arr = np.array(temp_list)
                
                candidates = [x for x in range(1, N + 1) if x not in temp_list]
                random.shuffle(candidates)
                
                # Try to add a candidate to temp
                swapped = False
                for x in candidates[:50]: # Check only 50 candidates per swap attempt
                    if _can_add_vectorized(x, temp_arr):
                        # Check if this swap is "better" or at least valid.
                        # Since we want max size, and size is same, we accept if it's valid.
                        # To avoid cycling, we might need a memory of visited states, but that's complex.
                        # Simple acceptance: accept if valid.
                        best = temp_list + [x]
                        best_arr = np.array(best)
                        improved = True
                        swapped = True
                        break
                
                if swapped:
                    break
                    
        if not improved:
            # If no add and no successful swap found in this iteration, 
            # we might be stuck. 
            # We can break or continue with random restarts of the local search phase?
            # The outer loop in construct() will handle restarts.
            break
            
    return best
=======
def _local_search(selected, time_limit):
    """Local search with 1-add, 1-swap, and remove-1-add-2 moves."""
    start = time.time()
    best = list(selected)
    best_size = len(best)
    best_arr = np.array(best)
    
    # Track which elements are in the set for fast lookup
    in_set = set(best)
    
    while time.time() - start < time_limit:
        improved = False
        
        # 1. Try to add any element
        if len(best) < MAX_SIZE:
            candidates = [x for x in range(1, N + 1) if x not in in_set]
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                if time.time() - start > time_limit:
                    break
                if _can_add_vectorized(x, best_arr):
                    best.append(x)
                    in_set.add(x)
                    best_arr = np.array(best)
                    improved = True
                    added = True
                    break
            
            if added:
                continue # Restart search with new set
        
        # 2. Try remove-1-add-2 move (key improvement for escaping local minima)
        # Remove one element, then try to add two new elements
        if len(best) >= 1 and len(best) < MAX_SIZE:
            # Try removing each element (or a random subset for speed)
            remove_indices = list(range(len(best)))
            random.shuffle(remove_indices)
            
            # Limit the number of removals to try per iteration
            max_removals_to_try = min(5, len(remove_indices))
            
            for idx in remove_indices[:max_removals_to_try]:
                if time.time() - start > time_limit:
                    break
                
                removed_elem = best[idx]
                temp_list = best[:idx] + best[idx+1:]
                temp_arr = np.array(temp_list)
                temp_in_set = in_set - {removed_elem}
                
                # Try to add two elements to temp
                candidates = [x for x in range(1, N + 1) if x not in temp_in_set]
                random.shuffle(candidates)
                
                added_count = 0
                new_elements = []
                current_list = temp_list
                current_arr = temp_arr
                current_in_set = set(temp_in_set)
                
                for x in candidates:
                    if time.time() - start > time_limit:
                        break
                    if _can_add_vectorized(x, current_arr):
                        current_list.append(x)
                        current_in_set.add(x)
                        current_arr = np.array(current_list)
                        new_elements.append(x)
                        added_count += 1
                        if added_count == 2:
                            break
                
                # Accept if we added 2 elements (net gain of 1) or 1 element (same size, might help future)
                if added_count == 2:
                    best = current_list
                    in_set = current_in_set
                    best_arr = current_arr
                    improved = True
                    break
                elif added_count == 1 and random.random() < 0.3:
                    # Occasionally accept single additions to explore different regions
                    best = current_list
                    in_set = current_in_set
                    best_arr = current_arr
                    improved = True
                    break
        
        # 3. Try 1-swap (remove one, add one) as fallback
        if not improved and len(best) > 0:
            for _ in range(10):
                if time.time() - start > time_limit:
                    break
                i = random.randint(0, len(best) - 1)
                removed_elem = best[i]
                temp_list = best[:i] + best[i+1:]
                temp_arr = np.array(temp_list)
                temp_in_set = in_set - {removed_elem}
                
                candidates = [x for x in range(1, N + 1) if x not in temp_in_set]
                random.shuffle(candidates)
                
                swapped = False
                for x in candidates[:50]:
                    if _can_add_vectorized(x, temp_arr):
                        best = temp_list + [x]
                        in_set = temp_in_set | {x}
                        best_arr = np.array(best)
                        improved = True
                        swapped = True
                        break
                
                if swapped:
                    break
                    
        if not improved:
            break
            
    return best
>>>>>>> REPLACE

