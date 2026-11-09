import time
import random

def construct():
    L = 232792560  # LCM of 1..22
    weights = {}
    for i in range(1, 23):
        weights[i] = L // i
    
    start_time = time.time()
    deadline = start_time + 80  # 80s margin
    
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]  # fallback size 9
    
    def check_collision(sums, w):
        """Check if adding weight w causes a collision with existing sums."""
        # New sums would be {s + w for s in sums}
        # Collision if any s + w is already in sums
        for s in sums:
            if s + w in sums:
                return True
        return False
    
    def backtrack(current_sums, current_set, remaining, target_size):
        """Try to build a set of target_size by adding elements from remaining."""
        nonlocal best_set
        
        if time.time() > deadline:
            return
        
        if len(current_set) == target_size:
            if len(current_set) > len(best_set):
                best_set = list(current_set)
            return
        
        # Pruning: not enough elements left
        if len(current_set) + len(remaining) < target_size:
            return
        
        # Try each remaining element
        for idx, elem in enumerate(remaining):
            if time.time() > deadline:
                return
            
            w = weights[elem]
            if not check_collision(current_sums, w):
                # Add this element
                new_sums = set(current_sums)
                for s in current_sums:
                    new_sums.add(s + w)
                
                backtrack(new_sums, current_set + [elem], remaining[idx+1:], target_size)
                
                # Early exit if we found a solution of target size
                if len(best_set) >= target_size:
                    return
    
    # Try sizes from 12 down to 10
    for target_size in [12, 11, 10]:
        if len(best_set) >= target_size:
            break
        
        # Try different orderings to find a solution
        # Order 1: descending by weight (larger elements first)
        elems_desc = sorted(range(1, 23), key=lambda x: -weights[x])
        backtrack({0}, [], elems_desc, target_size)
        
        if len(best_set) >= target_size:
            break
        
        if time.time() > deadline:
            break
        
        # Order 2: ascending by weight
        elems_asc = sorted(range(1, 23), key=lambda x: weights[x])
        backtrack({0}, [], elems_asc, target_size)
        
        if len(best_set) >= target_size:
            break
        
        if time.time() > deadline:
            break
        
        # Order 3: random shuffles
        for attempt in range(20):
            if time.time() > deadline:
                break
            elems_rand = list(range(1, 23))
            random.shuffle(elems_rand)
            backtrack({0}, [], elems_rand, target_size)
            if len(best_set) >= target_size:
                break
    
    return best_set
