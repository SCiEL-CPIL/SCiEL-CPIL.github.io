import time

N = 256

def construct():
    start_time = time.time()
    deadline = start_time + 25.0  # Safety margin for 30s limit
    
    selected = []
    used_sums = set()
    available = set(range(1, N + 1))
    
    # Pre-compute 2*x for quick access? Not strictly necessary.
    
    while available:
        if time.time() > deadline:
            break
            
        # 1. Identify valid candidates from 'available'
        # A candidate x is valid if:
        # - 2*x is not in used_sums
        # - (x + a) is not in used_sums for all a in selected
        valid_candidates = []
        
        for x in available:
            if 2 * x in used_sums:
                continue
            conflict = False
            for a in selected:
                if x + a in used_sums:
                    conflict = True
                    break
            if not conflict:
                valid_candidates.append(x)
                
        if not valid_candidates:
            break
            
        # 2. If only one valid candidate, pick it
        if len(valid_candidates) == 1:
            best_x = valid_candidates[0]
        else:
            # 3. Greedy Max-Blocking:
            # For each valid candidate x, count how many other available elements y
            # would become invalid if x were added.
            best_x = None
            max_blocks = -1
            
            for x in valid_candidates:
                # Simulate adding x
                # New sums added: {2*x} U {x+a for a in selected}
                new_sums = {2 * x}
                new_sums.update(x + a for a in selected)
                
                # Check which other available y are blocked
                blocks = 0
                for y in available:
                    if y == x:
                        continue
                    # Check if y is valid given selected U {x}
                    # y is invalid if:
                    # - 2*y in used_sums U new_sums
                    # - (y+a) in used_sums U new_sums for any a in selected U {x}
                    
                    # Optimization: We can just check against the union of sums
                    # But checking against the set union is O(|S|).
                    # Let's just check the conditions directly.
                    
                    if 2 * y in used_sums or 2 * y in new_sums:
                        continue # y is already blocked or newly blocked? 
                                 # Wait, if 2*y in used_sums, y was already invalid.
                                 # We only count y that were VALID before but are now INVALID.
                                 # So we should only consider y that were valid for 'selected'.
                                 # But we are iterating over 'available'.
                                 # 'available' contains elements that were valid for previous steps?
                                 # No, 'available' just contains unselected elements.
                                 # Some elements in 'available' might already be invalid for 'selected'.
                                 # We should only count y that are currently valid for 'selected' 
                                 # but become invalid for 'selected' U {x}.
                                 
                    # Let's refine: 
                    # First, determine which y in available are currently valid for 'selected'.
                    # This is expensive if done inside the loop.
                    
                    # Alternative: 
                    # Just count how many y in available \ {x} are NOT valid for selected U {x}.
                    # This includes y that were already invalid. That's fine, it's a proxy for "blocking power".
                    # Actually, it's better to count *newly* blocked valid elements.
                    
                    # Let's stick to a simpler metric: 
                    # Count how many y in available \ {x} conflict with x or x's sums.
                    
                    # A simpler and effective heuristic:
                    # Pick the x that minimizes the number of remaining valid candidates.
                    # Which is equivalent to maximizing the number of blocked candidates.
                    
                    pass # We will implement the counting in the loop below
                    
                # Let's re-structure the inner loop for efficiency
                # We want to maximize the number of y in available\{x} that are invalid for S U {x}
                # among those that were valid for S.
                
                # To avoid O(N^2 * |S|) complexity inside the candidate loop, 
                # we can just count total invalids.
                
                count_invalid = 0
                for y in available:
                    if y == x:
                        continue
                    # Check if y is valid for S U {x}
                    # If not, it's blocked.
                    if 2 * y in used_sums or 2 * y in new_sums:
                        count_invalid += 1
                        continue
                    conflict_y = False
                    for a in selected:
                        if y + a in used_sums:
                            conflict_y = True
                            break
                    if not conflict_y:
                        # Check against x
                        if y + x in used_sums or y + x in new_sums:
                            conflict_y = True
                        else:
                            # y+x is a new sum, so no conflict with used_sums.
                            # But we must ensure y+x doesn't conflict with other new sums?
                            # No, Sidon property is pairwise.
                            # If y+x is not in used_sums, and 2y not in used_sums,
                            # and y+a not in used_sums for a in S,
                            # then y is valid for S U {x} IF y+x is not equal to any other sum in S U {x}.
                            # The sums in S U {x} are used_sums U {2x} U {x+a}.
                            # We already checked y+a for a in S against used_sums.
                            # We need to check if y+x is in {2x} U {x+a} U used_sums.
                            # y+x == 2x => y==x (excluded)
                            # y+x == x+a => y==a (a is in S, y is in available, so y!=a)
                            # y+x in used_sums? Checked.
                            # So if y+x not in used_sums, y is valid.
                            
                            pass # y is valid
                            
                    if conflict_y:
                        count_invalid += 1
                        # Optimization: break early? No, we need the count.
                        
                if count_invalid > max_blocks:
                    max_blocks = count_invalid
                    best_x = x
            
        # Add best_x
        selected.append(best_x)
        used_sums.add(2 * best_x)
        for a in selected:
            if a != best_x:
                used_sums.add(best_x + a)
        
        # Remove best_x from available
        available.remove(best_x)
        
        # Optional: Prune available to only contain valid candidates for the new 'selected'
        # This speeds up future iterations
        new_available = set()
        for y in available:
            if 2 * y in used_sums:
                continue
            valid = True
            for a in selected:
                if y + a in used_sums:
                    valid = False
                    break
            if valid:
                new_available.add(y)
        available = new_available

    return selected

# EVOLVE-BLOCK-END
