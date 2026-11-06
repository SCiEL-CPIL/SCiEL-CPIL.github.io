"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
def construct():
    n = 256
    
    # 1. Greedy Construction
    # We maintain a boolean array for O(1) membership checks
    in_set = [False] * (n + 1)
    selected = []
    
    # Helper to check if x can be added to current set defined by in_set
    def can_add(x, in_set, selected):
        # Check if adding x creates a 3-AP
        # A 3-AP is a, b, c such that a+c=2b.
        # If we add x, it participates in a 3-AP if:
        # 1. x is the middle: exists a, b in S such that a+b=2x.
        # 2. x is an end: exists a in S such that 2x-a in S (and 2x-a != x).
        #    Or exists b in S such that 2b-x in S (and 2b-x != x).
        
        # Iterate through existing elements
        for y in selected:
            # Case 1: x is middle. y is one end. The other end is 2x - y.
            other = 2 * x - y
            if other > 0 and other <= n and other != x:
                if in_set[other]:
                    return False
            
            # Case 2: x is an end. y is the other end or middle.
            # If y is middle, the other end is 2y - x.
            other2 = 2 * y - x
            if other2 > 0 and other2 <= n and other2 != x:
                if in_set[other2]:
                    return False
            
            # Note: The case where x is an end and y is the *other* end is covered by 
            # "x is middle" if we swap roles? No.
            # Let's re-verify.
            # Triples involving x: {x, a, b}.
            # 2*x = a+b (x is middle)
            # 2*a = x+b (a is middle, b is other end) -> b = 2a - x. Check if b in S.
            # 2*b = x+a (b is middle, a is other end) -> a = 2b - x. Check if a in S.
            
            # My loop above:
            # for y in selected:
            #   other = 2*x - y. If in_set[other], then {y, x, other} is AP with x middle. Correct.
            #   other2 = 2*y - x. If in_set[other2], then {x, y, other2} is AP with y middle. Correct.
            #   Do we need to check if y is an end and x is the other end?
            #   That would mean {y, x, z} where y is end? No, if y and x are ends, z is middle.
            #   That is covered by "z is middle" where z is in S.
            #   Wait, if z is in S, and we check for z in the loop, we check 2*z - x.
            #   So yes, all cases are covered by iterating y in S and checking:
            #   1. Is 2*x - y in S? (x is middle)
            #   2. Is 2*y - x in S? (y is middle)
            #   Wait, what if y is an end and x is an end? Then the middle is (x+y)/2.
            #   If (x+y)/2 is in S, then we have an AP.
            #   Does my check catch this?
            #   If m = (x+y)/2 is in S, then when the loop reaches m (as y),
            #   it checks 2*m - x = y. Since y is in S, it returns False.
            #   So yes, it is covered.
        return True

    for x in range(1, n + 1):
        if can_add(x, in_set, selected):
            in_set[x] = True
            selected.append(x)
    
    # 2. Local Search Optimization
    # We want to maximize sum(1/x).
    # Current sum
    current_sum = sum(1.0 / x for x in selected)
    
    improved = True
    while improved:
        improved = False
        
        # Try to add any number not in set
        # Sort candidates by value (smallest first) to prioritize high reciprocal weight
        candidates = [x for x in range(1, n + 1) if not in_set[x]]
        candidates.sort()
        
        for x in candidates:
            if can_add(x, in_set, selected):
                in_set[x] = True
                selected.append(x)
                current_sum += 1.0 / x
                improved = True
                # Restart candidate check? Or just continue?
                # Adding x might invalidate future additions, but we check can_add each time.
                # However, adding x might make a previously invalid x' valid? No, adding constraints.
                # So we just continue.
        
        # Try to swap: remove y, add x where 1/x > 1/y (x < y)
        # We iterate y in descending order (largest y first, smallest 1/y)
        # and x in ascending order (smallest x first, largest 1/x)
        
        # To be efficient, we can just try all swaps where x < y
        # But O(N^2) swaps * O(K) check is N^2 * K. 256^2 * 100 = 6.5M ops. OK.
        
        # Let's do a pass of swaps
        # We need to be careful: removing y changes the set.
        # We should probably do this in a loop until no improvement.
        
        # Sort selected to iterate
        # We want to try removing large y.
        # Let's create a list of indices in selected corresponding to values
        # Or just iterate values
        
        # Efficient swap check:
        # For each y in selected (sorted descending):
        #   Temporarily remove y
        #   For each x not in set (sorted ascending, x < y):
        #      If can_add(x, set without y):
        #         If 1/x > 1/y:
        #            Swap!
        
        # To avoid O(N^2) inner loop if K is large, we can break early if x is large enough
        # such that 1/x < 1/y. Since x < y is required for gain, we only check x < y.
        
        # Let's implement this carefully.
        
        # We need a list of current elements
        # We will modify in_set and selected in place
        
        # To ensure we don't miss improvements, we might need multiple passes.
        # The 'improved' flag handles this.
        
        # Optimization: Precompute list of non-selected numbers
        non_selected = [x for x in range(1, n + 1) if not in_set[x]]
        non_selected.sort()
        
        # Iterate y from largest to smallest
        # We need to iterate over a copy or indices because we modify selected
        # But we only remove y and add x. 
        # If we swap, we break and restart the whole while loop?
        # Or we can just mark improved and continue.
        # If we swap, the set changes, so the 'non_selected' list is stale.
        # So it's safer to do one swap, set improved=True, and break out of loops to restart the while loop.
        
        y_list = sorted(selected, reverse=True)
        
        for y in y_list:
            if not in_set[y]: # Might have been swapped out in a previous iteration? No, we break on swap.
                continue
                
            # Temporarily remove y
            in_set[y] = False
            # We don't remove from selected list yet to keep it simple, 
            # but can_add uses 'selected' list. 
            # Problem: can_add checks 'selected'. If y is still in 'selected' but in_set[y]=False,
            # can_add might behave incorrectly if it relies on selected membership.
            # My can_add implementation uses 'selected' for iteration and 'in_set' for lookup.
            # If y is in 'selected' but in_set[y] is False, it will still be iterated.
            # This is a bug risk.
            
            # Better: Remove y from selected list properly.
            # But removing from list is O(K).
            # Let's just use a set for membership in can_add? No, we need iteration.
            
            # Let's refine can_add to take a set or list that is consistent.
            # I will remove y from selected, call can_add, then if swap fails, put it back.
            
            # Find index of y in selected
            try:
                idx_y = selected.index(y)
            except ValueError:
                in_set[y] = True
                continue
            
            selected.pop(idx_y)
            
            # Now try to add x
            swapped = False
            for x in non_selected:
                if x >= y:
                    break # No gain possible if x >= y (1/x <= 1/y)
                
                if can_add(x, in_set, selected):
                    # Check gain
                    if 1.0 / x > 1.0 / y:
                        # Perform swap
                        in_set[x] = True
                        selected.append(x)
                        current_sum += 1.0 / x - 1.0 / y
                        improved = True
                        swapped = True
                        break
                    else:
                        # Since non_selected is sorted ascending, if 1/x <= 1/y, 
                        # all subsequent x' > x will have 1/x' < 1/x <= 1/y.
                        # So we can break?
                        # Wait, we want 1/x > 1/y.
                        # If x is small, 1/x is large.
                        # non_selected is sorted ascending: 1, 2, 3...
                        # So 1/x is descending.
                        # If 1/x <= 1/y, then for all larger x, 1/x is even smaller.
                        # So yes, break.
                        break
            
            if swapped:
                # We modified in_set and selected.
                # We need to update non_selected? 
                # We break out of the while loop anyway.
                break
            
            # If no swap, restore y
            in_set[y] = True
            selected.append(y)
            # Re-insert at correct position? Order doesn't matter for can_add logic, 
            # but for 'selected.index' in next iterations it does.
            # Actually, can_add just iterates. Order doesn't matter.
            # But 'selected.index' in the next y-iteration might be slow if list is large? O(K).
            # It's fine.
            
        if improved:
            # Restart to recalculate non_selected and check for more additions
            continue
            
    return selected
# EVOLVE-BLOCK-END
