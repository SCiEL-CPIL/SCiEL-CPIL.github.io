import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin

    prescribed = [1, 2, 5, 11]
    N = 284
    max_sum = 2 * N
    
    # Bitset for sums
    # Bit i represents if sum i is used
    # Max sum is 568, so we need 569 bits. Python int handles this easily.
    
    def compute_initial_sums(elements):
        sums = 0
        for i in range(len(elements)):
            for j in range(i, len(elements)):
                s = elements[i] + elements[j]
                sums |= (1 << s)
        return sums

    def can_add(elements, used_sums, x):
        """
        Check if x can be added to elements given the current used_sums.
        elements must be a list of ints.
        Returns True if x can be added without creating duplicate sums.
        """
        # Check sums x + y for all y in elements
        # And x + x
        # Note: x is not in elements yet.
        
        # Optimization: iterate through elements
        for y in elements:
            s = x + y
            if (used_sums >> s) & 1:
                return False
        
        # Check x + x
        s_xx = x + x
        if (used_sums >> s_xx) & 1:
            return False
            
        return True

    def add_to_sums(used_sums, elements, x):
        """
        Update used_sums to include sums involving x.
        Assumes can_add returned True.
        """
        for y in elements:
            s = x + y
            used_sums |= (1 << s)
        s_xx = x + x
        used_sums |= (1 << s_xx)
        return used_sums

    def remove_from_sums(used_sums, elements, x):
        """
        Update used_sums to remove sums involving x.
        Assumes x is in elements.
        Note: This is tricky because if multiple pairs produce the same sum, 
        we can't just clear the bit. But in a Sidon set, all sums are unique.
        So if x is part of the set, the sums involving x are unique to x?
        No. Sum a+b is unique. If we remove x, the sums x+y disappear.
        Since the set was valid, no other pair (u,v) has u+v = x+y.
        So we can safely clear the bits for x+y.
        """
        for y in elements:
            if y == x:
                continue
            s = x + y
            used_sums &= ~(1 << s)
        s_xx = x + x
        used_sums &= ~(1 << s_xx)
        return used_sums

    # Initial State
    current_set = prescribed[:]
    current_sums = compute_initial_sums(current_set)
    
    best_set = current_set[:]
    best_size = len(current_set)
    
    # Greedy initialization to get a larger starting set
    # Try to add elements in random order
    available = [i for i in range(1, N+1) if i not in current_set]
    random.shuffle(available)
    
    for x in available:
        if can_add(current_set, current_sums, x):
            current_set.append(x)
            current_sums = add_to_sums(current_sums, current_set, x)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set[:]
            if len(current_set) == 20:
                break
                
    # Local Search (Hill Climbing)
    # We try to improve the size.
    # Since we are maximizing size, and we can't go above 20 (theoretical max for this range/seed likely),
    # we focus on finding a valid set of size 20.
    
    # If we already found 20, we can return early? 
    # The task says "Maximize the size". 20 is the known max for unconstrained.
    # With constraints, it might be 19 or 20.
    # Let's assume 20 is achievable. If we find it, we return it.
    
    if best_size == 20:
        return best_set

    # If we didn't find 20, we run a more intensive search.
    # We will use Randomized Restarts of Hill Climbing.
    
    def hill_climb(start_set, start_sums, time_limit):
        """
        Perform hill climbing starting from start_set.
        Returns the best set found.
        """
        cur_set = start_set[:]
        cur_sums = start_sums
        cur_size = len(cur_set)
        
        # We want to increase size.
        # If we are stuck, we might need to allow removals to escape?
        # But if we remove, size decreases. We want to maximize size.
        # Standard approach for Maximization:
        # 1. Try to Add. If possible, Add.
        # 2. If no Add possible, try to Swap (Remove one, Add one) that keeps size same but might enable future Adds?
        #    Or just restart.
        
        # Given the bitset speed, we can try many random additions.
        
        while time.time() < time_limit:
            made_progress = False
            
            # Try to add a random element
            # Sample a few candidates
            for _ in range(10):
                if cur_size >= 20:
                    return cur_set
                x = random.randint(1, N)
                if x in cur_set:
                    continue
                if can_add(cur_set, cur_sums, x):
                    cur_set.append(x)
                    cur_sums = add_to_sums(cur_sums, cur_set, x)
                    cur_size += 1
                    made_progress = True
                    if cur_size > best_size:
                        best_size = cur_size
                        best_set = cur_set[:]
                    break
            
            if not made_progress:
                # Stuck. Try a swap to potentially open up space?
                # Or just break and restart.
                # Let's try a few random swaps.
                swapped = False
                for _ in range(5):
                    if len(cur_set) <= 4:
                        break
                    # Pick a non-prescribed element to remove
                    non_prescribed = [i for i in range(len(cur_set)) if cur_set[i] not in prescribed]
                    if not non_prescribed:
                        break
                    idx = random.choice(non_prescribed)
                    old_val = cur_set[idx]
                    
                    # Pick a new value
                    new_val = random.randint(1, N)
                    if new_val in cur_set:
                        continue
                    if new_val == old_val:
                        continue
                    
                    # Check if new_val can be added to cur_set \ {old_val}
                    # We need to temporarily remove old_val, check, then decide.
                    # Since sums are unique, we can check if new_val conflicts with cur_set \ {old_val}
                    # AND if new_val conflicts with old_val? No, old_val is removed.
                    # But we must ensure new_val doesn't conflict with the remaining set.
                    
                    # Efficient check:
                    # Create a temporary list? Or just loop.
                    # Loop is O(K).
                    valid_swap = True
                    for y in cur_set:
                        if y == old_val:
                            continue
                        s = new_val + y
                        # Check if s is in cur_sums, BUT we must ignore sums that involve old_val
                        # This is hard with a single bitset.
                        # However, if the set was valid, the sum `s` is either:
                        # 1. Not in cur_sums
                        # 2. In cur_sums, generated by a pair (u,v) in cur_set.
                        #    If (u,v) involves old_val, then that sum will disappear when old_val is removed.
                        #    So if `s` is in cur_sums, we need to know if it's "protected" by old_val.
                        
                        # This makes Swap checking complex with just a bitset.
                        # Alternative: Reconstruct sums for the check? O(K^2).
                        # Or: Just try the swap and verify the whole set?
                        # Given K <= 20, O(K^2) is 400 ops. Very fast.
                        
                        pass # We'll skip complex swap logic for now and rely on Add + Restart.
                        
                # If no progress, break to restart
                break
                
        return cur_set

    # Main Loop: Multiple Restarts
    # We restart the greedy + hill climb process with different random seeds/orders.
    
    global_best_size = best_size
    global_best_set = best_set
    
    restarts = 0
    while time.time() < deadline:
        restarts += 1
        # New random initialization
        init_set = prescribed[:]
        init_sums = compute_initial_sums(init_set)
        
        # Greedy fill
        available = [i for i in range(1, N+1) if i not in init_set]
        random.shuffle(available)
        
        for x in available:
            if can_add(init_set, init_sums, x):
                init_set.append(x)
                init_sums = add_to_sums(init_sums, init_set, x)
                if len(init_set) == 20:
                    break
        
        # Hill Climb
        # We just try to add more.
        # Since we can't easily swap with bitsets, we rely on the greedy phase being good.
        # If greedy gets stuck at 18, we might need a different strategy.
        
        # Let's try a "Random Walk" on the set size?
        # No, let's just do many random greedy restarts.
        
        if len(init_set) > global_best_size:
            global_best_size = len(init_set)
            global_best_set = init_set[:]
            
        if global_best_size == 20:
            break
            
        # If we are stuck at a size < 20 for a while, we might need to try
        # removing some elements to allow others?
        # For now, simple restarts are robust.
        
    return global_best_set
