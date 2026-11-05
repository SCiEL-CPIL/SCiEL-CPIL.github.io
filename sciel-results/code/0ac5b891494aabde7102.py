import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    prescribed = [1, 2, 5, 11]
    prescribed_set = frozenset(prescribed)
    best_size = 0
    best_set = set(prescribed_set)
    
    # Pre-compute sums for prescribed set
    def get_sums(s):
        sums = set()
        lst = sorted(s)
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                sums.add(lst[i] + lst[j])
        return sums

    initial_sums = get_sums(prescribed_set)
    
    # Helper to check if x can be added to a set with given sums
    def can_add(x, s_set, s_sums):
        if x in s_set:
            return False
        # Check x + y for all y in s_set
        for y in s_set:
            if x + y in s_sums:
                return False
        # Check 2x
        if 2 * x in s_sums:
            return False
        return True

    # Helper to update sums when adding x
    def add_to_sums(x, s_set, s_sums):
        new_sums = set(s_sums)
        for y in s_set:
            new_sums.add(x + y)
        new_sums.add(2 * x)
        return new_sums

    # Local search: try to add elements, then try swaps more aggressively
    def local_search(current_set, current_sums, max_steps=300):
        steps = 0
        while steps < max_steps and time.time() < deadline:
            steps += 1
            # Try to add any element
            added = False
            candidates_list = list(range(12, 285))
            random.shuffle(candidates_list)
            
            for x in candidates_list:
                if can_add(x, current_set, current_sums):
                    current_set.add(x)
                    current_sums = add_to_sums(x, current_set, current_sums)
                    added = True
                    break
            
            if added:
                continue
            
            # If no element could be added, try a swap more aggressively
            removable = [x for x in current_set if x not in prescribed_set]
            if not removable:
                break
                
            # Try all removals systematically
            swap_done = False
            for x_remove in removable:
                if time.time() >= deadline:
                    break
                    
                temp_set = set(current_set)
                temp_set.remove(x_remove)
                temp_sums = get_sums(temp_set)
                
                # Try all possible additions
                for y_add in range(1, 285):
                    if y_add in temp_set:
                        continue
                    if can_add(y_add, temp_set, temp_sums):
                        # Accept swap only if it keeps size same or improves
                        # Since we're swapping 1 for 1, size stays same
                        # But we want to enable future additions, so accept
                        current_set = temp_set
                        current_set.add(y_add)
                        current_sums = add_to_sums(y_add, current_set, temp_sums)
                        swap_done = True
                        break
                
                if swap_done:
                    break
            
            if not swap_done:
                # Stuck - try a more drastic move: remove two, add one or two
                if len(removable) >= 2:
                    # Try removing two elements and adding up to two
                    for i in range(len(removable)):
                        if time.time() >= deadline:
                            break
                        for j in range(i+1, len(removable)):
                            if time.time() >= deadline:
                                break
                            temp_set = set(current_set)
                            temp_set.remove(removable[i])
                            temp_set.remove(removable[j])
                            temp_sums = get_sums(temp_set)
                            
                            # Try to add one or two elements
                            added_count = 0
                            new_sums = temp_sums
                            for y_add in range(1, 285):
                                if y_add in temp_set:
                                    continue
                                if can_add(y_add, temp_set, new_sums):
                                    temp_set.add(y_add)
                                    new_sums = add_to_sums(y_add, temp_set, new_sums)
                                    added_count += 1
                                    if added_count >= 2:
                                        break
                            
                            # Accept if we added at least as many as we removed
                            if added_count >= 2:
                                current_set = temp_set
                                current_sums = new_sums
                                swap_done = True
                                break
                        if swap_done:
                            break
                
                if not swap_done:
                    # Give up on this local search
                    break

        return current_set, current_sums

    # Main search loop
    # We will do several random greedy restarts and local searches.
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        
        # 1. Random Greedy Construction with multiple strategies
        current_set = set(prescribed_set)
        current_sums = get_sums(current_set)
        
        # Strategy 1: Random order
        if num_restarts % 3 == 0:
            candidates_list = list(range(12, 285))
            random.shuffle(candidates_list)
        # Strategy 2: Ascending order (prefer smaller values to leave room)
        elif num_restarts % 3 == 1:
            candidates_list = list(range(12, 285))
        # Strategy 3: Descending order
        else:
            candidates_list = list(range(284, 11, -1))
        
        for x in candidates_list:
            if can_add(x, current_set, current_sums):
                current_set.add(x)
                current_sums = add_to_sums(x, current_set, current_sums)
        
        # 2. Local Search with more steps
        current_set, current_sums = local_search(current_set, current_sums, max_steps=300)
        
        # 3. Update Best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            
            # Optional: If we hit a very high size, we might stop?
            # Max is 20. If we hit 20, it's likely optimal.
            if best_size >= 20:
                break

    # Final polish: exhaustive single swaps on the best set
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            removable = [x for x in best_set if x not in prescribed_set]
            for x_remove in removable:
                if time.time() >= deadline - 5:
                    break
                temp_set = set(best_set)
                temp_set.remove(x_remove)
                temp_sums = get_sums(temp_set)
                
                # Try all possible additions
                for y_add in range(1, 285):
                    if y_add in temp_set:
                        continue
                    if can_add(y_add, temp_set, temp_sums):
                        # Accept any valid swap to explore the space
                        best_set = temp_set
                        best_set.add(y_add)
                        improved = True
                        break
                if improved:
                    break
        
        # After swaps, try to add more elements
        best_sums = get_sums(best_set)
        for x in range(12, 285):
            if time.time() >= deadline - 2:
                break
            if can_add(x, best_set, best_sums):
                best_set.add(x)
                best_sums = add_to_sums(x, best_set, best_sums)

    # Final validation before returning
    final_sums = get_sums(best_set)
    if len(final_sums) != len(best_set) * (len(best_set) + 1) // 2:
        # Fallback to prescribed set if something went wrong
        return sorted(prescribed)
        
    return sorted(list(best_set))
