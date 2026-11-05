import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin for 120s limit
    
    fixed = {1, 2, 5, 11}
    best_set = list(fixed)
    best_size = len(fixed)
    
    # Pre-calculate initial sums
    def get_sums(s):
        sums = set()
        lst = list(s)
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                sums.add(lst[i] + lst[j])
        return sums
    
    def is_valid_addition(x, current_set, current_sums):
        for y in current_set:
            if x + y in current_sums:
                return False
        if x + x in current_sums:
            return False
        return True
    
    def add_to_set(x, current_set, current_sums):
        for y in current_set:
            current_sums.add(x + y)
        current_sums.add(x + x)
        current_set.append(x)
        current_set.sort()
    
    def remove_from_set(x, current_set, current_sums):
        for y in current_set:
            if y != x:
                current_sums.discard(x + y)
        current_sums.discard(x + x)
        current_set.remove(x)
    
    # Phase 1: Branch and Bound to get a strong initial solution
    candidates = [x for x in range(1, 285) if x not in fixed]
    current_set = list(fixed)
    current_sums = get_sums(current_set)
    
    def backtrack(idx, cur_set, cur_sums, cur_size):
        nonlocal best_set, best_size
        
        if time.time() > deadline - 30:  # Leave 30s for local search
            return
            
        # Pruning: if we can't beat best, stop
        remaining = len(candidates) - idx
        if cur_size + remaining <= best_size:
            return
            
        # Base case
        if idx == len(candidates):
            if cur_size > best_size:
                best_size = cur_size
                best_set = list(cur_set)
            return
            
        x = candidates[idx]
        
        # Try to include x
        valid = True
        new_sums = set()
        for y in cur_set:
            s = x + y
            if s in cur_sums:
                valid = False
                break
            new_sums.add(s)
        
        if valid and (x + x) in cur_sums:
            valid = False
        elif valid:
            new_sums.add(x + x)
        
        if valid:
            cur_set.append(x)
            cur_sums.update(new_sums)
            backtrack(idx + 1, cur_set, cur_sums, cur_size + 1)
            cur_set.pop()
            for s in new_sums:
                cur_sums.discard(s)
        
        # Skip x
        backtrack(idx + 1, cur_set, cur_sums, cur_size)
    
    # Run B&B with a time limit
    bb_deadline = start_time + 20  # 20s for B&B
    original_deadline = deadline
    deadline = bb_deadline
    backtrack(0, current_set, current_sums, len(current_set))
    deadline = original_deadline
    
    # Update best if B&B found something better
    if len(current_set) > best_size:
        best_size = len(current_set)
        best_set = list(current_set)
    
    # Phase 2: Destroy and Repair local search
    # Start from the best set found
    current_set = list(best_set)
    current_sums = get_sums(current_set)
    
    attempts = 0
    no_improve_count = 0
    max_no_improve = 50
    
    while time.time() < deadline - 5:  # Leave 5s buffer
        attempts += 1
        
        # Destroy: remove 1-3 random non-fixed elements
        removable = [x for x in current_set if x not in fixed]
        if not removable:
            break
            
        num_remove = min(len(removable), random.randint(1, 3))
        to_remove = random.sample(removable, num_remove)
        
        for x in to_remove:
            remove_from_set(x, current_set, current_sums)
        
        # Repair: try to add elements back or new ones
        # Create candidates: all not in current_set
        all_candidates = [x for x in range(1, 285) if x not in current_set]
        random.shuffle(all_candidates)
        
        for x in all_candidates:
            if time.time() > deadline - 5:
                break
            if is_valid_addition(x, current_set, current_sums):
                add_to_set(x, current_set, current_sums)
        
        # Check for improvement
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = list(current_set)
            no_improve_count = 0
            
            # If we hit the theoretical max, stop
            if best_size >= 20:
                break
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Try a different strategy: swap-based local search
                if time.time() > deadline - 10:
                    break
                # Do a few swap operations
                for _ in range(5):
                    if time.time() > deadline - 5:
                        break
                    # Pick a random element to remove and try to find a replacement
                    if not removable:
                        break
                    x_remove = random.choice(removable)
                    remove_from_set(x_remove, current_set, current_sums)
                    
                    # Try to add any valid element
                    candidates_now = [x for x in range(1, 285) if x not in current_set]
                    random.shuffle(candidates_now)
                    added = False
                    for x_add in candidates_now:
                        if is_valid_addition(x_add, current_set, current_sums):
                            add_to_set(x_add, current_set, current_sums)
                            added = True
                            break
                    
                    if not added:
                        # Put back what we removed
                        add_to_set(x_remove, current_set, current_sums)
                    
                    if len(current_set) > best_size:
                        best_size = len(current_set)
                        best_set = list(current_set)
                        no_improve_count = 0
                        if best_size >= 20:
                            break
                    
                    removable = [x for x in current_set if x not in fixed]
                
                no_improve_count = 0  # Reset after swap phase
    
    return best_set
