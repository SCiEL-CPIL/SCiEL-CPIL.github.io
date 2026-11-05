import time
import random

def construct():
    start_time = time.time()
    deadline = 85.0 # 90s limit, leave 5s margin
    
    # 1. Two-Phase Greedy Initialization
    # Phase 1: Greedy on [51, 100]
    # Phase 2: Greedy on [1, 50]
    selected = []
    reachable = 1
    
    # Phase 1
    for x in range(100, 50, -1):
        new_reach = reachable | (reachable << x)
        if not (new_reach >> 200) & 1:
            selected.append(x)
            reachable = new_reach
            
    # Phase 2
    for x in range(50, 0, -1):
        new_reach = reachable | (reachable << x)
        if not (new_reach >> 200) & 1:
            selected.append(x)
            reachable = new_reach
            
    best_set = set(selected)
    best_size = len(best_set)
    
    # 2. Local Search with Directed Moves
    current_set = best_set.copy()
    current_reachable = reachable
    
    all_elements = list(range(1, 101))
    
    def can_add(x, R):
        return not ((R >> (200 - x)) & 1)
        
    def update_reachable(R, x):
        return R | (R << x)
        
    def is_valid_set(S):
        r = 1
        for x in S:
            r |= (r << x)
            if (r >> 200) & 1:
                return False
        return True

    improved = True
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any single missing element
        missing = [x for x in all_elements if x not in current_set]
        random.shuffle(missing)
        
        for x in missing:
            if can_add(x, current_reachable):
                current_set.add(x)
                current_reachable = update_reachable(current_reachable, x)
                improved = True
                if len(current_set) > best_size:
                    best_set = current_set.copy()
                    best_size = len(current_set)
        
        # Phase 2: Directed Swaps
        if not improved:
            # Try 1-1 swaps to explore neighborhood
            # This helps escape local optima where no single addition is possible
            # but a swap enables a future addition
            
            # Optimization: Precompute list of current and missing
            curr_list = list(current_set)
            miss_list = [x for x in all_elements if x not in current_set]
            
            # 1-1 Swaps
            found_11 = False
            for a in curr_list:
                if time.time() > deadline: break
                for b in miss_list:
                    # Check if S - {a} + {b} is valid
                    # We can compute reachable for S-{a} quickly? 
                    # No, we have to recompute or use a trick.
                    # Recomputing is O(|S|).
                    # Let's just check validity of the new set directly.
                    new_set = current_set.copy()
                    new_set.remove(a)
                    new_set.add(b)
                    
                    if is_valid_set(new_set):
                        current_set = new_set
                        # Recalculate reachable
                        current_reachable = 1
                        for x in current_set:
                            current_reachable |= (current_reachable << x)
                        improved = True
                        found_11 = True
                        if len(current_set) > best_size:
                            best_set = current_set.copy()
                            best_size = len(current_set)
                        break
                if found_11 or time.time() > deadline: break
            
            if found_11:
                continue # Restart loop to try additions again

            # Try 1-2 swaps to increase size
            # Remove 1, Add 2
            found_12 = False
            for a in curr_list:
                if time.time() > deadline: break
                # Sample pairs from missing to keep it fast, or iterate all if small enough
                # If miss_list is large, iterate all pairs is expensive.
                # Let's iterate all pairs but break early if found.
                for i in range(len(miss_list)):
                    if time.time() > deadline: break
                    b = miss_list[i]
                    for j in range(i+1, len(miss_list)):
                        if time.time() > deadline: break
                        c = miss_list[j]
                        
                        new_set = current_set.copy()
                        new_set.remove(a)
                        new_set.add(b)
                        new_set.add(c)
                        
                        if is_valid_set(new_set):
                            current_set = new_set
                            current_reachable = 1
                            for x in current_set:
                                current_reachable |= (current_reachable << x)
                            improved = True
                            found_12 = True
                            if len(current_set) > best_size:
                                best_set = current_set.copy()
                                best_size = len(current_set)
                            break
                    if found_12 or time.time() > deadline: break
                if found_12 or time.time() > deadline: break
            
            if found_12:
                continue # Restart loop to try additions again

    return list(best_set)
