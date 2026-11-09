import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 5s safety margin
    
    fixed = [1, 2, 5, 11]
    best_set = set(fixed)
    best_size = len(best_set)
    
    # Helper to compute sums of a set
    def get_sums(s):
        sums = set()
        lst = list(s)
        for i in range(len(lst)):
            for j in range(i, len(lst)):
                sums.add(lst[i] + lst[j])
        return sums
        
    # Helper to check if x can be added to s
    def can_add(x, s, sums):
        for y in s:
            if x + y in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    # Helper to add x to s and update sums
    def add_element(x, s, sums):
        s.add(x)
        for y in s:
            if y != x: # y < x or y > x, just add all pairs involving x
                sums.add(x + y)
        sums.add(2 * x)
        
    # Helper to remove x from s and update sums
    def remove_element(x, s, sums):
        s.remove(x)
        # This is expensive if we just recompute sums.
        # Given small size, recomputing sums for the whole set is O(k^2) ~ 400 ops.
        # Doing this 1000 times is 400k ops. Very fast.
        # So we can just recompute sums after a swap.
        pass

    # Strategy:
    # 1. Randomized Greedy
    # 2. Local Search (Swap + Refill)
    
    while time.time() < deadline:
        current_set = set(fixed)
        sums = get_sums(current_set)
        
        # Phase 1: Randomized Greedy
        cands = [x for x in range(1, 285) if x not in current_set]
        random.shuffle(cands)
        
        for x in cands:
            if can_add(x, current_set, sums):
                add_element(x, current_set, sums)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = set(current_set)
            if best_size >= 20:
                break
                
        # Phase 2: Local Search
        # Try to improve by swapping
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try a few random swaps
            for _ in range(50): # Limit swaps per iteration to avoid getting stuck
                if time.time() > deadline:
                    break
                non_fixed = [x for x in current_set if x not in fixed]
                if not non_fixed:
                    break
                x = random.choice(non_fixed)
                y = random.choice([z for z in range(1, 285) if z not in current_set])
                
                # Check if swap is valid
                # Remove x, add y
                temp_set = set(current_set)
                temp_set.remove(x)
                temp_set.add(y)
                temp_sums = get_sums(temp_set)
                
                if len(temp_sums) == len(temp_set) * (len(temp_set) + 1) // 2:
                    # Valid swap
                    # Check if we can add more
                    old_size = len(current_set)
                    current_set = temp_set
                    sums = temp_sums
                    
                    # Try to add more
                    cands2 = [z for z in range(1, 285) if z not in current_set]
                    random.shuffle(cands2)
                    for z in cands2:
                        if can_add(z, current_set, sums):
                            add_element(z, current_set, sums)
                            
                    if len(current_set) > old_size:
                        improved = True
                        if len(current_set) > best_size:
                            best_size = len(current_set)
                            best_set = set(current_set)
                            if best_size >= 20:
                                break
                    # If no improvement, we might keep the swap if it doesn't hurt, 
                    # but to avoid drifting, let's only keep if it leads to improvement 
                    # or just revert? 
                    # Actually, keeping valid swaps can help escape local minima.
                    # But if we don't improve, we might just be wandering.
                    # Let's revert if no improvement to stay close to the greedy result?
                    # Or keep it? 
                    # Let's keep it if it's valid, but mark as "no improvement" for this loop step.
                    # Wait, the loop condition is `improved`. 
                    # If we swap and don't improve, `improved` stays False, so we break the inner while.
                    # This means we only do one round of swaps per outer iteration.
                    # That's fine.
                    
            if best_size >= 20:
                break
                
    return list(best_set)
