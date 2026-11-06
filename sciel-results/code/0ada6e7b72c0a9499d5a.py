import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    fixed = [1, 2, 5, 11]
    fixed_set = set(fixed)
    max_val = 284
    
    # Max sum is 284 + 284 = 568.
    # We use a Python integer as a bitset.
    
    def get_initial_mask(elems):
        mask = 0
        for i in range(len(elems)):
            for j in range(i, len(elems)):
                mask |= (1 << (elems[i] + elems[j]))
        return mask

    def is_valid(x, mask, elems):
        # Check if x can be added to elems
        # 1. Check x + x
        if (mask >> (x + x)) & 1:
            return False
        # 2. Check x + y for all y in elems
        for y in elems:
            if (mask >> (x + y)) & 1:
                return False
        return True

    def get_new_mask(x, mask, elems):
        new_mask = mask | (1 << (x + x))
        for y in elems:
            new_mask |= (1 << (x + y))
        return new_mask

    def greedy_build(seed):
        random.seed(seed)
        current_elems = list(fixed)
        current_mask = get_initial_mask(current_elems)
        
        candidates = [x for x in range(1, max_val + 1) if x not in fixed_set]
        random.shuffle(candidates)
        
        for x in candidates:
            if is_valid(x, current_mask, current_elems):
                current_elems.append(x)
                current_mask = get_new_mask(x, current_mask, current_elems[:-1])
                
        return current_elems, current_mask

    def local_search(initial_elems, initial_mask, seed):
        random.seed(seed)
        best_elems = list(initial_elems)
        best_mask = initial_mask
        best_size = len(initial_elems)
        
        current_elems = list(initial_elems)
        current_mask = initial_mask
        current_size = len(initial_elems)
        
        no_improve_count = 0
        max_no_improve = 1000
        
        while time.time() < deadline - 2:
            if no_improve_count > max_no_improve:
                break
                
            improved = False
            
            # 1. Try to add a valid element
            # Sort candidates by some heuristic? Random is fine.
            candidates = [x for x in range(1, max_val + 1) if x not in current_elems]
            random.shuffle(candidates)
            
            added = False
            for x in candidates:
                if is_valid(x, current_mask, current_elems):
                    current_elems.append(x)
                    current_mask = get_new_mask(x, current_mask, current_elems[:-1])
                    current_size += 1
                    added = True
                    improved = True
                    break
            
            if not added:
                # 2. Try a swap: remove one, add one
                # Pick a random element to remove (not fixed)
                removable = [y for y in current_elems if y not in fixed_set]
                if removable:
                    y_to_remove = random.choice(removable)
                    # Temporarily remove
                    old_elems = list(current_elems)
                    old_mask = current_mask
                    current_elems.remove(y_to_remove)
                    # Recalculate mask (expensive, but n is small)
                    current_mask = get_initial_mask(current_elems)
                    current_size -= 1
                    
                    # Try to add a new element
                    candidates = [x for x in range(1, max_val + 1) if x not in current_elems]
                    random.shuffle(candidates)
                    
                    swapped = False
                    for x in candidates:
                        if is_valid(x, current_mask, current_elems):
                            current_elems.append(x)
                            current_mask = get_new_mask(x, current_mask, current_elems[:-1])
                            current_size += 1
                            swapped = True
                            break
                    
                    if not swapped:
                        # Restore
                        current_elems = old_elems
                        current_mask = old_mask
                        current_size += 1
                    else:
                        improved = True
                else:
                    # Can't remove fixed elements, stuck
                    break

            if improved:
                no_improve_count = 0
                if current_size > best_size:
                    best_size = current_size
                    best_elems = list(current_elems)
                    best_mask = current_mask
            else:
                no_improve_count += 1
                
                # 3. If stuck, perform a kick: remove a random element
                # This helps escape local optima
                removable = [y for y in current_elems if y not in fixed_set]
                if removable:
                    y_to_remove = random.choice(removable)
                    current_elems.remove(y_to_remove)
                    current_mask = get_initial_mask(current_elems)
                    current_size -= 1
                else:
                    break
                    
        return best_elems, best_mask

    best_overall = list(fixed)
    best_mask_overall = get_initial_mask(best_overall)
    best_size = len(best_overall)
    
    # Run multiple iterations of (Greedy + Local Search)
    # We have ~115s. Each LS run should be limited to allow many restarts.
    # Let's say we try 20 restarts.
    
    num_restarts = 30
    time_per_restart = (deadline - time.time() - 5) / num_restarts
    
    for i in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        # Greedy phase
        elems, mask = greedy_build(i)
        
        if len(elems) > best_size:
            best_size = len(elems)
            best_overall = elems
            best_mask_overall = mask
            
        # Local Search phase
        # Allocate remaining time for this restart
        ls_deadline = time.time() + time_per_restart
        # Pass a modified deadline? 
        # The local_search function uses global deadline. 
        # Let's refactor to pass deadline.
        
        # For now, we'll just call local_search. 
        # To prevent it from running too long, we can modify the logic inside 
        # or just trust that it breaks on no_improve.
        
        # Actually, let's just run local_search with a tighter internal deadline 
        # if we were to refactor. Here, we rely on the global deadline 
        # and the no_improve counter.
        
        # To make it safer, we can check time in the main loop.
        
        # Let's just run one local search per greedy.
        final_elems, final_mask = local_search(elems, mask, 1000 + i)
        
        if len(final_elems) > best_size:
            best_size = len(final_elems)
            best_overall = final_elems
            best_mask_overall = final_mask
            
    return best_overall
