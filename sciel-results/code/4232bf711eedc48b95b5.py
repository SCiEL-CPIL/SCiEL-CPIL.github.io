"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50 seconds safety margin
    
    def is_valid_sums(sums_set):
        """Check if a set of subset sums is valid (no divisibility). Assumes no duplicates."""
        if not sums_set:
            return True
        max_sum = max(sums_set)
        for a in sums_set:
            for k in range(2, max_sum // a + 1):
                if (a * k) in sums_set:
                    return False
        return True

    def try_add(x, current_sums):
        """
        Attempts to add x to the set represented by current_sums.
        Returns (True, new_sums) if valid, (False, None) otherwise.
        """
        # Generate new sums
        # New sums are {x} U {s+x | s in current_sums} U current_sums
        # We need to check:
        # 1. No duplicates in the new set.
        # 2. No divisibility in the new set.
        
        # Optimization: 
        # If current_sums is valid, we only need to check:
        # - x against current_sums
        # - x against s+x
        # - s+x against current_sums
        # - s+x against t+x (for s != t)
        
        # Simple approach: build the set and check.
        # Since size is small (< 2^10), this is fast enough.
        
        new_sums = set(current_sums)
        new_sums.add(x)
        for s in current_sums:
            new_sums.add(s + x)
            
        # Check duplicates
        # The number of new sums added is 1 (for x) + len(current_sums) (for s+x)
        # Total expected size = len(current_sums) + 1 + len(current_sums) = 2*len(current_sums) + 1
        if len(new_sums) != 2 * len(current_sums) + 1:
            return False, None
            
        # Check divisibility
        # We can just run the full check on new_sums
        if not is_valid_sums(new_sums):
            return False, None
            
        return True, new_sums

    def local_search():
        # Start with a valid set
        # Try to build a set greedily first to get a decent start
        current_elements = []
        current_sums = set()
        
        # Greedy initialization: try adding large numbers first
        candidates = list(range(256, 0, -1))
        random.shuffle(candidates)
        
        for x in candidates:
            ok, new_sums = try_add(x, current_sums)
            if ok:
                current_elements.append(x)
                current_sums = new_sums
                
        best_elements = list(current_elements)
        best_sums = set(current_sums)
        
        # Local Search Phase
        # We will try to swap elements or add/remove to increase size
        # We focus on increasing size. If size increases, we keep it.
        # If size is same, we might keep it to explore (annealing), but for now let's just hill climb.
        
        num_iterations = 0
        while time.time() < deadline - 2:
            num_iterations += 1
            if num_iterations % 100 == 0 and time.time() > deadline:
                break
                
            # Randomly pick a move
            # 1. Try to add a random element
            # 2. Try to swap a random element with another
            # 3. Try to remove an element (only if we can add 2 later? Or just to unblock?)
            
            # Strategy: Try to add a random element not in the set
            if random.random() < 0.7:
                # Try adding
                available = [x for x in range(1, 257) if x not in current_elements]
                if not available:
                    continue
                x_add = random.choice(available)
                ok, new_sums = try_add(x_add, current_sums)
                if ok:
                    current_elements.append(x_add)
                    current_sums = new_sums
                    if len(current_elements) > len(best_elements):
                        best_elements = list(current_elements)
                        best_sums = set(current_sums)
            else:
                # Try swapping
                if not current_elements:
                    continue
                idx = random.randint(0, len(current_elements) - 1)
                x_remove = current_elements[idx]
                
                # Temporarily remove
                # Recalculating sums after removal is expensive.
                # Instead, let's just restart greedy if we get stuck?
                # Or, we can try to replace x_remove with a new x_add.
                
                # To do this efficiently, we need the sums of the set WITHOUT x_remove.
                # This is hard to maintain incrementally.
                # Let's stick to "Add" moves primarily, and occasionally "Restart" if stuck.
                
                # If we are stuck (no adds possible), we should probably perturb.
                # But "Add" is the only move that increases size.
                # If no add is possible, we are at a local maximum.
                # We can try to remove one element and see if we can add two.
                
                # Let's implement a simple "Remove and Re-add"
                # Remove x_remove
                # We need to compute sums of remaining elements.
                # This is O(2^k). k is small.
                remaining_elements = [e for i, e in enumerate(current_elements) if i != idx]
                temp_sums = set()
                for e in remaining_elements:
                    new_sums_temp = set()
                    for s in temp_sums:
                        new_sums_temp.add(s + e)
                    new_sums_temp.add(e)
                    temp_sums.update(new_sums_temp)
                
                # Now try to add a new element to this reduced set
                available = [x for x in range(1, 257) if x not in remaining_elements]
                random.shuffle(available)
                improved = False
                for x_add in available:
                    ok, new_sums = try_add(x_add, temp_sums)
                    if ok:
                        # We added one. Can we add another?
                        # Let's just take the first one that increases size relative to original?
                        # Original size was len(current_elements).
                        # New size is len(remaining_elements) + 1 = len(current_elements).
                        # So size is same.
                        # We need to check if we can add MORE.
                        
                        # Let's just update current state to this new valid set of same size.
                        # This allows exploration.
                        current_elements = remaining_elements + [x_add]
                        current_sums = new_sums
                        
                        # Check if we can add another to this new state
                        available2 = [x for x in range(1, 257) if x not in current_elements]
                        random.shuffle(available2)
                        for x_add2 in available2:
                            ok2, new_sums2 = try_add(x_add2, current_sums)
                            if ok2:
                                current_elements.append(x_add2)
                                current_sums = new_sums2
                                if len(current_elements) > len(best_elements):
                                    best_elements = list(current_elements)
                                    best_sums = set(current_sums)
                                improved = True
                                break
                        if improved:
                            break
                        
                        # If we didn't add a second one, we have a set of same size.
                        # We keep it to explore further.
                        if len(current_elements) > len(best_elements):
                             best_elements = list(current_elements)
                             best_sums = set(current_sums)
                        break
                # If no swap/add worked, we stay in current state (or maybe revert? 
                # Since we modified current_elements, we should probably revert if no improvement found?
                # But for local search, staying in a different valid state of same size is good for exploration.
                # However, if we just swapped x_remove with x_add, and couldn't add more, 
                # we are in a new local minimum/maximum.
                
    result = local_search()
    
    # Fallback: if local search fails or returns empty, do a simple greedy
    if not result:
        current_elements = []
        current_sums = set()
        for x in range(256, 0, -1):
            ok, new_sums = try_add(x, current_sums)
            if ok:
                current_elements.append(x)
                current_sums = new_sums
        result = current_elements
        
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
